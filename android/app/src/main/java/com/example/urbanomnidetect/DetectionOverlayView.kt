package com.example.urbanomnidetect

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import java.util.Locale
import kotlin.math.min

class DetectionOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var detections: List<Detection> = emptyList()
    private var imageWidth = 0
    private var imageHeight = 0

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f
    }
    private val cuboidPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1.5f
    }
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val labelBackgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 14f * resources.displayMetrics.density
        isFakeBoldText = true
    }

    fun setImageSize(width: Int, height: Int) {
        imageWidth = width
        imageHeight = height
        invalidate()
    }

    fun setDetections(nextDetections: List<Detection>) {
        detections = nextDetections
        invalidate()
    }

    fun clearDetections() {
        detections = emptyList()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (imageWidth <= 0 || imageHeight <= 0 || width <= 0 || height <= 0) return

        val scale = min(width.toFloat() / imageWidth, height.toFloat() / imageHeight)
        val offsetX = (width - imageWidth * scale) / 2f
        val offsetY = (height - imageHeight * scale) / 2f

        detections.forEach { detection ->
            val color = classColor(detection.classId)
            boxPaint.color = color
            cuboidPaint.color = color
            fillPaint.color = Color.argb(40, Color.red(color), Color.green(color), Color.blue(color))
            labelBackgroundPaint.color =
                Color.argb(210, Color.red(color), Color.green(color), Color.blue(color))

            val rect = RectF(
                offsetX + detection.box.left * scale,
                offsetY + detection.box.top * scale,
                offsetX + detection.box.right * scale,
                offsetY + detection.box.bottom * scale
            )
            canvas.drawRect(rect, fillPaint)
            canvas.drawRect(rect, boxPaint)

            if (detection.cuboidCorners.size == CUBOID_CORNER_COUNT) {
                val corners = detection.cuboidCorners.map { point ->
                    PointF(offsetX + point.x * scale, offsetY + point.y * scale)
                }
                val groundPath = polygonPath(corners, 0)
                canvas.drawPath(groundPath, fillPaint)
                drawEdges(canvas, corners, GROUND_EDGES)
                drawEdges(canvas, corners, TOP_EDGES)
                drawEdges(canvas, corners, VERTICAL_EDGES)
            }

            val label = "${COCO_NAMES.getOrElse(detection.classId) { "class ${detection.classId}" }} " +
                String.format(Locale.US, "%.2f", detection.confidence)
            val padding = 4f * resources.displayMetrics.density
            val labelHeight = labelPaint.textSize + padding * 2
            val labelWidth = labelPaint.measureText(label) + padding * 2
            val labelTop = (rect.top - labelHeight).coerceAtLeast(offsetY)
            canvas.drawRect(
                rect.left,
                labelTop,
                rect.left + labelWidth,
                labelTop + labelHeight,
                labelBackgroundPaint
            )
            canvas.drawText(label, rect.left + padding, labelTop + labelHeight - padding, labelPaint)
        }
    }

    private fun drawEdges(canvas: Canvas, points: List<PointF>, edges: List<Pair<Int, Int>>) {
        edges.forEach { (from, to) ->
            canvas.drawLine(
                points[from].x,
                points[from].y,
                points[to].x,
                points[to].y,
                cuboidPaint
            )
        }
    }

    private fun polygonPath(points: List<PointF>, startIndex: Int): Path =
        Path().apply {
            moveTo(points[startIndex].x, points[startIndex].y)
            for (index in startIndex + 1..startIndex + 3) {
                lineTo(points[index].x, points[index].y)
            }
            close()
        }

    private fun classColor(classId: Int): Int = when (classId) {
        0 -> Color.rgb(255, 168, 64) // person
        1, 3 -> Color.rgb(181, 99, 232) // bicycle, motorcycle
        2 -> Color.rgb(64, 156, 255) // car
        5 -> Color.rgb(54, 200, 130) // bus
        7 -> Color.rgb(255, 99, 99) // truck
        else -> Color.rgb(150, 162, 176)
    }

    private companion object {
        const val CUBOID_CORNER_COUNT = 8
        val GROUND_EDGES = listOf(0 to 1, 1 to 2, 2 to 3, 3 to 0)
        val TOP_EDGES = listOf(4 to 5, 5 to 6, 6 to 7, 7 to 4)
        val VERTICAL_EDGES = listOf(0 to 4, 1 to 5, 2 to 6, 3 to 7)
        val COCO_NAMES = listOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
            "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
            "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
            "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        )
    }
}

data class Detection(
    val classId: Int,
    val confidence: Float,
    val box: RectF,
    val cuboidCorners: List<PointF> = emptyList()
)
