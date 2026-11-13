package com.example.urbanomnidetect

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.PointF
import android.graphics.RectF
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class UrbanOmniDetectRunner(
    private val context: Context,
    private val modelAsset: String = "urbanomnidetect_v2_x_640.onnx",
    private val inputSize: Int = 640,
    private val confidenceThreshold: Float = 0.1f,
    private val iouThreshold: Float = 0.7f,
    private val keypointThreshold: Float = 0.5f
) : AutoCloseable {

    private val environment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null

    val isReady: Boolean
        get() = session != null

    fun initialize() {
        if (session != null) return

        val modelFile = File(context.cacheDir, modelAsset)
        if (!modelFile.exists()) {
            val temporaryFile = File(context.cacheDir, "$modelAsset.part")
            try {
                context.assets.open(modelAsset).use { input ->
                    FileOutputStream(temporaryFile).use { output -> input.copyTo(output) }
                }
                check(temporaryFile.renameTo(modelFile)) {
                    "Could not move the copied ONNX model into the app cache."
                }
            } finally {
                if (temporaryFile.exists()) temporaryFile.delete()
            }
        }
        session = environment.createSession(modelFile.absolutePath)
    }

    fun runInference(bitmap: Bitmap): List<Detection> {
        val ortSession = checkNotNull(session) { "ONNX model has not been initialized." }
        require(bitmap.width > 0 && bitmap.height > 0) { "Input image is empty." }

        val letterboxed = letterbox(bitmap)
        val inputName = ortSession.inputNames.firstOrNull()
            ?: error("The ONNX model has no input tensor.")
        val inputShape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        val inputBuffer = ByteBuffer.allocateDirect(letterboxed.rgbNchw.size * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(letterboxed.rgbNchw)
            .apply { rewind() }
        val inputTensor = OnnxTensor.createTensor(
            environment,
            inputBuffer,
            inputShape
        )

        inputTensor.use { tensor ->
            ortSession.run(mapOf(inputName to tensor)).use { result ->
                val output = result.firstOrNull()?.value as? OnnxTensor
                    ?: error("The ONNX model returned no tensor output.")
                val tensorInfo = output.info as? TensorInfo
                    ?: error("The model output is not a tensor.")
                val outputData = FloatArray(output.floatBuffer.remaining()).also {
                    output.floatBuffer.get(it)
                }
                return decodeOutput(
                    outputData = outputData,
                    outputShape = tensorInfo.shape,
                    transform = letterboxed.transform,
                    sourceWidth = bitmap.width,
                    sourceHeight = bitmap.height
                )
            }
        }
    }

    override fun close() {
        session?.close()
        session = null
    }

    private fun letterbox(bitmap: Bitmap): LetterboxedImage {
        val scale = min(inputSize.toFloat() / bitmap.width, inputSize.toFloat() / bitmap.height)
        val resizedWidth = (bitmap.width * scale).roundToInt()
        val resizedHeight = (bitmap.height * scale).roundToInt()
        val padLeft = (inputSize - resizedWidth) / 2
        val padTop = (inputSize - resizedHeight) / 2
        val resized = Bitmap.createScaledBitmap(bitmap, resizedWidth, resizedHeight, true)
        val canvasBitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        Canvas(canvasBitmap).apply {
            drawColor(Color.rgb(114, 114, 114))
            drawBitmap(resized, padLeft.toFloat(), padTop.toFloat(), null)
        }
        if (resized !== bitmap) resized.recycle()

        val rgbNchw = FloatArray(3 * inputSize * inputSize)
        val argb = IntArray(inputSize * inputSize)
        canvasBitmap.getPixels(argb, 0, inputSize, 0, 0, inputSize, inputSize)
        val planeSize = inputSize * inputSize
        for (i in argb.indices) {
            rgbNchw[i] = Color.red(argb[i]) / 255f
            rgbNchw[planeSize + i] = Color.green(argb[i]) / 255f
            rgbNchw[2 * planeSize + i] = Color.blue(argb[i]) / 255f
        }
        canvasBitmap.recycle()

        return LetterboxedImage(
            rgbNchw = rgbNchw,
            transform = LetterboxTransform(scale, padLeft.toFloat(), padTop.toFloat())
        )
    }

    private fun decodeOutput(
        outputData: FloatArray,
        outputShape: LongArray,
        transform: LetterboxTransform,
        sourceWidth: Int,
        sourceHeight: Int
    ): List<Detection> {
        require(outputShape.size == 3 && outputShape[0] == 1L) {
            "Unsupported ONNX output shape ${outputShape.contentToString()}; expected batch size 1 and rank 3."
        }

        val dimA = outputShape[1].toInt()
        val dimB = outputShape[2].toInt()
        require(dimA > 0 && dimB > 0 && outputData.size == dimA * dimB) {
            "ONNX output data size does not match shape ${outputShape.contentToString()}."
        }

        // Ultralytics YOLO pose exports commonly expose either raw [1, 4+80+24, anchors]
        // predictions or end-to-end [1, detections, 6+24] rows.
        val rawChannels = COCO_CLASS_COUNT + BOX_CHANNELS + KEYPOINT_COUNT * KEYPOINT_CHANNELS
        val rawChannelsFirst = dimA == rawChannels
        val rawChannelsLast = dimB == rawChannels
        val decodedChannels = 6 + KEYPOINT_COUNT * KEYPOINT_CHANNELS
        val decodedChannelsLast = dimB == decodedChannels
        val decodedChannelsFirst = dimA == decodedChannels

        return when {
            rawChannelsFirst -> decodeRaw(
                outputData, dimA, dimB, true, transform, sourceWidth, sourceHeight
            )
            rawChannelsLast -> decodeRaw(
                outputData, dimB, dimA, false, transform, sourceWidth, sourceHeight
            )
            decodedChannelsLast -> decodeEndToEnd(
                outputData, dimA, dimB, false, transform, sourceWidth, sourceHeight
            )
            decodedChannelsFirst -> decodeEndToEnd(
                outputData, dimB, dimA, true, transform, sourceWidth, sourceHeight
            )
            else -> error(
                "Unsupported ONNX output shape ${outputShape.contentToString()}. " +
                    "Expected raw pose channels $rawChannels or end-to-end rows $decodedChannels."
            )
        }
    }

    private fun decodeRaw(
        data: FloatArray,
        channels: Int,
        candidates: Int,
        channelsFirst: Boolean,
        transform: LetterboxTransform,
        sourceWidth: Int,
        sourceHeight: Int
    ): List<Detection> {
        fun value(channel: Int, candidate: Int): Float =
            if (channelsFirst) data[channel * candidates + candidate]
            else data[candidate * channels + channel]

        val predictions = ArrayList<Detection>()
        for (candidate in 0 until candidates) {
            var classId = 0
            var confidence = Float.NEGATIVE_INFINITY
            for (classIndex in 0 until COCO_CLASS_COUNT) {
                val score = value(BOX_CHANNELS + classIndex, candidate)
                if (score > confidence) {
                    confidence = score
                    classId = classIndex
                }
            }
            if (!confidence.isFinite() || confidence < confidenceThreshold) continue

            val centerX = value(0, candidate)
            val centerY = value(1, candidate)
            val boxWidth = value(2, candidate)
            val boxHeight = value(3, candidate)
            val box = mapBox(
                centerX - boxWidth / 2f, centerY - boxHeight / 2f,
                centerX + boxWidth / 2f, centerY + boxHeight / 2f,
                transform, sourceWidth, sourceHeight
            ) ?: continue

            val points = ArrayList<PointF>(KEYPOINT_COUNT)
            val visibility = FloatArray(KEYPOINT_COUNT)
            for (keypoint in 0 until KEYPOINT_COUNT) {
                val offset = BOX_CHANNELS + COCO_CLASS_COUNT + keypoint * KEYPOINT_CHANNELS
                points += PointF(
                    transform.toSourceX(value(offset, candidate)).coerceIn(0f, sourceWidth.toFloat()),
                    transform.toSourceY(value(offset + 1, candidate)).coerceIn(0f, sourceHeight.toFloat())
                )
                visibility[keypoint] = value(offset + 2, candidate)
            }

            predictions += Detection(
                classId = classId,
                confidence = confidence,
                box = box,
                cuboidCorners = points.takeIf {
                    classId in ROAD_USER_CLASSES &&
                    isValidCuboid(it, visibility, box)
                } ?: emptyList()
            )
        }
        return classAwareNms(predictions)
    }

    private fun decodeEndToEnd(
        data: FloatArray,
        rows: Int,
        columns: Int,
        channelsFirst: Boolean,
        transform: LetterboxTransform,
        sourceWidth: Int,
        sourceHeight: Int
    ): List<Detection> {
        fun value(row: Int, column: Int): Float =
            if (channelsFirst) data[column * rows + row]
            else data[row * columns + column]

        val detections = ArrayList<Detection>()
        for (row in 0 until rows) {
            val confidence = value(row, 4)
            val classId = value(row, 5).roundToInt()
            if (!confidence.isFinite() || confidence < confidenceThreshold ||
                classId !in 0 until COCO_CLASS_COUNT
            ) continue

            val box = mapBox(
                value(row, 0), value(row, 1), value(row, 2), value(row, 3),
                transform, sourceWidth, sourceHeight
            ) ?: continue
            val points = ArrayList<PointF>(KEYPOINT_COUNT)
            val visibility = FloatArray(KEYPOINT_COUNT)
            for (keypoint in 0 until KEYPOINT_COUNT) {
                val offset = 6 + keypoint * KEYPOINT_CHANNELS
                points += PointF(
                    transform.toSourceX(value(row, offset)).coerceIn(0f, sourceWidth.toFloat()),
                    transform.toSourceY(value(row, offset + 1)).coerceIn(0f, sourceHeight.toFloat())
                )
                visibility[keypoint] = value(row, offset + 2)
            }

            detections += Detection(
                classId = classId,
                confidence = confidence,
                box = box,
                cuboidCorners = points.takeIf {
                    classId in ROAD_USER_CLASSES && isValidCuboid(it, visibility, box)
                } ?: emptyList()
            )
        }
        // The end-to-end model already performs its own NMS-free top-k selection.
        return detections.sortedByDescending { it.confidence }
    }

    private fun isValidCuboid(
        points: List<PointF>,
        visibility: FloatArray,
        box: RectF
    ): Boolean {
        if (points.size < KEYPOINT_COUNT || visibility.size < KEYPOINT_COUNT) return false
        if (visibility.any { !it.isFinite() || it < keypointThreshold }) return false
        val expanded = RectF(
            box.left - 0.75f * box.width(),
            box.top - 0.75f * box.height(),
            box.right + 0.75f * box.width(),
            box.bottom + 0.75f * box.height()
        )
        return points.all { point ->
            point.x.isFinite() && point.y.isFinite() &&
                point.x >= expanded.left && point.x <= expanded.right &&
                point.y >= expanded.top && point.y <= expanded.bottom &&
                (kotlin.math.abs(point.x) + kotlin.math.abs(point.y) > 1e-6f)
        }
    }

    private fun mapBox(
        x1: Float,
        y1: Float,
        x2: Float,
        y2: Float,
        transform: LetterboxTransform,
        sourceWidth: Int,
        sourceHeight: Int
    ): RectF? {
        val box = RectF(
            transform.toSourceX(x1).coerceIn(0f, sourceWidth.toFloat()),
            transform.toSourceY(y1).coerceIn(0f, sourceHeight.toFloat()),
            transform.toSourceX(x2).coerceIn(0f, sourceWidth.toFloat()),
            transform.toSourceY(y2).coerceIn(0f, sourceHeight.toFloat())
        )
        return box.takeIf { it.width() > 0f && it.height() > 0f }
    }

    private fun classAwareNms(detections: List<Detection>): List<Detection> {
        val ordered = detections.sortedByDescending { it.confidence }
        val kept = ArrayList<Detection>()
        for (candidate in ordered) {
            if (kept.none { it.classId == candidate.classId && iou(it.box, candidate.box) > iouThreshold }) {
                kept += candidate
                if (kept.size == MAX_DETECTIONS) break
            }
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val intersectionWidth = max(0f, min(a.right, b.right) - max(a.left, b.left))
        val intersectionHeight = max(0f, min(a.bottom, b.bottom) - max(a.top, b.top))
        val intersection = intersectionWidth * intersectionHeight
        val union = a.width() * a.height() + b.width() * b.height() - intersection
        return if (union > 0f) intersection / union else 0f
    }

    private data class LetterboxedImage(
        val rgbNchw: FloatArray,
        val transform: LetterboxTransform
    )

    private data class LetterboxTransform(
        val scale: Float,
        val padLeft: Float,
        val padTop: Float
    ) {
        fun toSourceX(x: Float): Float = (x - padLeft) / scale
        fun toSourceY(y: Float): Float = (y - padTop) / scale
    }

    private companion object {
        const val COCO_CLASS_COUNT = 80
        const val BOX_CHANNELS = 4
        const val KEYPOINT_COUNT = 8
        const val KEYPOINT_CHANNELS = 3
        const val MAX_DETECTIONS = 300
        val ROAD_USER_CLASSES = setOf(0, 1, 2, 3, 5, 7)
    }
}
