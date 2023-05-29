package com.example.faceapitest

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.media.ThumbnailUtils
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Size
import android.widget.Button
import android.widget.ImageView
import android.widget.PopupMenu
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import java.io.File

class MainActivity : AppCompatActivity() {
    var selectedFileUri: Uri? = null
    var isFileImage = true
    lateinit var imgView: ImageView
    var selectMediaLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == RESULT_OK) {
                selectedFileUri = result.data?.data
                //isFileImage = contentResolver.getType(selectedFileUri!!)!!.contains("image")
                if(isFileImage)
                    imgView.setImageURI(selectedFileUri)
                else{
                    val thumbnailSize = Size(300, 300)
                    val cancellationSignal = android.os.CancellationSignal()

                    val thumbnail = getFileFromUri(selectedFileUri!!)?.let {
                        ThumbnailUtils.createVideoThumbnail(
                            it, thumbnailSize, cancellationSignal)
                    }
                    imgView.setImageBitmap(thumbnail)
                }
            }
        }

    private fun getFileFromUri(uri: Uri): File? {
        val filePath: String = getRealPathFromURI(uri) ?: return null
        return File(filePath)
    }

    private fun getRealPathFromURI(uri: Uri): String? {
        val projection = arrayOf(MediaStore.Images.Media.DATA)
        val cursor = contentResolver.query(uri, projection, null, null, null)
        if (cursor != null) {
            val columnIndex = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA)
            cursor.moveToFirst()
            val filePath = cursor.getString(columnIndex)
            cursor.close()
            return filePath
        }
        return uri.path
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        supportActionBar?.title = ""
        imgView = findViewById<ImageView>(R.id.imageView)


        val requestPermissionLauncher =
            registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
                if (isGranted) {
                    showDialog()
                } else {
                    Toast.makeText(this, "Permission denied", Toast.LENGTH_SHORT).show()
                }
            }

        imgView.setOnClickListener {
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.READ_EXTERNAL_STORAGE
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                requestPermissionLauncher.launch(Manifest.permission.READ_EXTERNAL_STORAGE)
            } else {
                showDialog()
            }
        }

        val button = findViewById<Button>(R.id.button)
        button.setOnClickListener {
            if (selectedFileUri != null) {
                val intent = Intent(this, LoadingActivity::class.java)
                intent.putExtra("imageUri", selectedFileUri.toString())
                startActivity(intent)
                return@setOnClickListener
            }

            Toast.makeText(this, "Please select an image/video", Toast.LENGTH_SHORT).show()
        }
    }

    fun showDialog(){
        val options = arrayOf("Select Image", "Select Video")
        val builder = AlertDialog.Builder(this)
        builder.setTitle("Select Media")
        builder.setItems(options) {_, which->
            when(which) {
                0 -> {
                    val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
                    selectMediaLauncher.launch(intent)
                }
                1 -> {
                    val intent = Intent(Intent.ACTION_PICK, MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
                    selectMediaLauncher.launch(intent)
                }
            }
        }
        builder.create().show()
    }
}