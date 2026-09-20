package com.voice.shield.api

import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import okhttp3.OkHttpClient
import java.util.concurrent.TimeUnit

object RetrofitClient {
    private var currentBaseUrl: String = "http://10.78.43.212:8000/"
    private var apiInstance: TrustVoiceApi? = null

    fun getBaseUrl(): String = currentBaseUrl

    fun setBaseUrl(newUrl: String) {
        val trimmed = newUrl.trim()
        val formatted = if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
            trimmed
        } else {
            "http://$trimmed"
        }
        val finalized = if (formatted.endsWith("/")) formatted else "$formatted/"
        if (currentBaseUrl != finalized) {
            currentBaseUrl = finalized
            apiInstance = null
        }
    }

    val instance: TrustVoiceApi
        get() {
            if (apiInstance == null) {
                val client = OkHttpClient.Builder()
                    .connectTimeout(30, TimeUnit.SECONDS)
                    .readTimeout(30, TimeUnit.SECONDS)
                    .writeTimeout(30, TimeUnit.SECONDS)
                    .build()

                val retrofit = Retrofit.Builder()
                    .baseUrl(currentBaseUrl)
                    .addConverterFactory(GsonConverterFactory.create())
                    .client(client)
                    .build()

                apiInstance = retrofit.create(TrustVoiceApi::class.java)
            }
            return apiInstance!!
        }
}
