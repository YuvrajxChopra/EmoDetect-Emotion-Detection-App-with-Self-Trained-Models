package com.example.faceapitest

import android.content.Intent
import android.media.Image
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.ActionBar
import androidx.core.net.toUri
import org.json.JSONArray
import org.json.JSONObject
import androidx.core.content.ContextCompat
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import com.google.gson.Gson
import okhttp3.MediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody


data class OpenAIResponse(val choices: List<Choice>)
data class Choice(val text: String, val confidence: Float)

class ResultActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)
        val actionBar: ActionBar? = supportActionBar
        if (actionBar != null) {
            actionBar.setTitle("")
        }
        val responseBody = intent.getStringExtra("responseBody")
        val imageUri = intent.getStringExtra("imageUri")!!.toUri()

        findViewById<ImageView>(R.id.imageView).setImageURI(imageUri)

        val jsonObject = JSONObject(responseBody)

        val age = jsonObject.getString("age")
        val gender = jsonObject.getString("gender")
        val mainemotion = jsonObject.getString("main_emotion")
        var txt = ""

        if (gender == "Male") {
            txt = "He is "
        } else {
            txt = "She is "
        }

        txt += age + " years old and feels:\n"

        val emotions = jsonObject.names() ?: JSONArray()

        for (i in 0 until emotions.length()) {
            val emotion = emotions.getString(i)
            if (emotion != "age" && emotion != "gender" && emotion != "main_emotion") {
                val percentages = jsonObject.getString(emotion)
                var percentage = (percentages.toDouble() * 100).toInt()
                if (percentage == 0) {percentage = (percentages.toDouble() * 10000).toInt()}
                txt += "$emotion: $percentage\n"
            }
        }

        findViewById<TextView>(R.id.emotion_txt).text = txt


//
//        findViewById<TextView>(R.id.angerTxt).text = "$angerLikelihoodValue% Anger"
//        findViewById<TextView>(R.id.joyTxt).text = "$joyLikelihoodValue% Joy"
//        findViewById<TextView>(R.id.sorrowTxt).text = "$sorrowLikelihoodValue% Sorrow"
//        findViewById<TextView>(R.id.surpriseTxt).text = "$surpriseLikelihoodValue% Surprise"
//
//        findViewById<ProgressBar>(R.id.anger).progress = angerLikelihoodValue
//        findViewById<ProgressBar>(R.id.joy).progress = joyLikelihoodValue
//        findViewById<ProgressBar>(R.id.sorrow).progress = sorrowLikelihoodValue
//        findViewById<ProgressBar>(R.id.surprise).progress = surpriseLikelihoodValue

        var textView = findViewById<TextView>(R.id.openai)
        val prompt = "Respond in 5 or less words suggesting something to person who is feeling " + mainemotion
        val response = getOpenAIResponse(prompt)
        textView.text = response

        val backButton = findViewById<Button>(R.id.back_button)
        backButton.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
        }
    }
    private fun getOpenAIResponse(prompt: String): String? {
        val client = OkHttpClient()
        val url = "https://api.openai.com/v1/engines/davinci-codex/completions"
        val json = """
            {
                "prompt": "$prompt",
                "max_tokens": 100,
                "temperature": 0.5,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }
        """.trimIndent()
        val requestBody = json.toMediaTypeOrNull()?.let { mediaType ->
            RequestBody.create(mediaType, json)
        }
        val request = requestBody?.let {
            Request.Builder()
                .url(url)
                .header("Content-Type", "application/json")
                .header("Authorization", "sk-...jBLA")
                .post(it)
                .build()
        }
        val response = request?.let { client.newCall(it).execute() }
        val responseJson = response?.body?.string()
        val openAIResponse = Gson().fromJson(responseJson, OpenAIResponse::class.java)
        return openAIResponse?.choices?.get(0)?.text
    }
}