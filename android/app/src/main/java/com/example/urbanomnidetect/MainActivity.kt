package com.example.urbanomnidetect

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var overlayView: DetectionOverlayView
    private lateinit var runner: UrbanOmniDetectRunner
    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    private var selectedBitmap: Bitmap? = null
    @Volatile
    private var modelReady = false

    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) loadImage(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        overlayView = findViewById(R.id.overlayView)
        runner = UrbanOmniDetectRunner(this)

        findViewById<Button>(R.id.pickImageButton).setOnClickListener {
            pickImageLauncher.launch("image/*")
        }
        findViewById<Button>(R.id.runInferenceButton).setOnClickListener {
            runInference()
        }

        inferenceExecutor.execute {
            try {
                runner.initialize()
                modelReady = true
            } catch (error: Exception) {
                runOnUiThread {
                    Toast.makeText(
                        this,
                        "Could not load the ONNX model: ${error.message}",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }
    }

    override fun onDestroy() {
        if (::runner.isInitialized) {
            inferenceExecutor.execute { runner.close() }
        }
        inferenceExecutor.shutdown()
        super.onDestroy()
    }

    private fun loadImage(uri: Uri) {
        try {
            val bitmap = contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
                ?: error("The selected image could not be opened.")
            selectedBitmap = bitmap
            imageView.setImageBitmap(bitmap)
            overlayView.setImageSize(bitmap.width, bitmap.height)
            overlayView.clearDetections()
        } catch (error: Exception) {
            Toast.makeText(
                this,
                "Could not load image: ${error.message}",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    private fun runInference() {
        val bitmap = selectedBitmap
        if (bitmap == null) {
            Toast.makeText(this, "Please pick an image first.", Toast.LENGTH_SHORT).show()
            return
        }
        if (!modelReady) {
            Toast.makeText(
                this,
                "Model is not ready. Ensure urbanomnidetect_v2_x_640.onnx is in app/src/main/assets.",
                Toast.LENGTH_LONG
            ).show()
            return
        }

        findViewById<Button>(R.id.runInferenceButton).isEnabled = false
        inferenceExecutor.execute {
            try {
                val detections = runner.runInference(bitmap)
                runOnUiThread {
                    if (selectedBitmap === bitmap) overlayView.setDetections(detections)
                    findViewById<Button>(R.id.runInferenceButton).isEnabled = true
                }
            } catch (error: Exception) {
                runOnUiThread {
                    findViewById<Button>(R.id.runInferenceButton).isEnabled = true
                    Toast.makeText(
                        this,
                        "Inference failed: ${error.message}",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }
    }
}
