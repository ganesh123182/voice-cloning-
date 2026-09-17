package com.voice.shield.api

import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Body
import retrofit2.Response

import retrofit2.http.FormUrlEncoded
import retrofit2.http.Field
import retrofit2.http.Header


import retrofit2.http.Multipart
import retrofit2.http.Part
import okhttp3.MultipartBody
import okhttp3.RequestBody

data class DashboardResponse(
    val threats_blocked: Int,
    val recent_activity: List<ActivityItem>
)

data class ActivityItem(
    val id: String,
    val caller_name: String,
    val phone_number: String,
    val status: String, // "Safe", "Blocked", "Verified"
    val timestamp: String
)

data class EnrollmentResponse(
    val success: Boolean,
    val message: String,
    val user_id: String?,
    val evidence_hash: String?,
    val blockchain_tx_hash: String?,
    val enrollment_id: String?
)


data class AuthResponse(
    val message: String? = null,
    val access_token: String? = null,
    val token_type: String? = null,
    val user_id: String? = null,
    val username: String? = null,
    val full_name: String? = null
)

data class TokenResponse(val access_token: String, val token_type: String)
data class GenericResponse(val message: String, val user_id: String?)
data class EnrollmentStatusResponse(
    val enrollment_id: String,
    val status: String,
    val evidence_hash: String,
    val blockchain_tx_hash: String?,
    val created_at: String
)
data class VerificationResponse(
    val verified: Boolean,
    val reason: String?,
    val evidence_hash: String?,
    val blockchain_status: String?
)

data class DetectionScores(
    val fake_prob: Float?,
    val real_prob: Float?
)

data class DetectVoiceResponse(
    val success: Boolean,
    val prediction: String?,
    val confidence: Float?,
    val ai_probability: Float?,
    val risk_level: String?,
    val scores: DetectionScores?,
    val details: List<String>?,
    val explanation: List<String>?,
    val processing_time: Float?,
    val disclaimer: String?
)

interface TrustVoiceApi {
    @GET("/api/dashboard")
    suspend fun getDashboardData(): Response<DashboardResponse>

    @Multipart
    @POST("/api/enroll_voice")
    suspend fun enrollVoice(
        @Header("Authorization") token: String,
        @Part file: MultipartBody.Part
    ): Response<EnrollmentResponse>

    @Multipart
    @POST("/api/detect")
    suspend fun detectVoice(
        @Part file: MultipartBody.Part
    ): Response<DetectVoiceResponse>

    @FormUrlEncoded
    @POST("/api/login")
    suspend fun login(
        @Field("username") u: String,
        @Field("password") p: String
    ): Response<AuthResponse>

    @FormUrlEncoded
    @POST("/api/register")
    suspend fun register(
        @Field("username") u: String,
        @Field("password") p: String,
        @Field("full_name") fn: String? = null,
        @Field("email") em: String? = null,
        @Field("phone") ph: String? = null
    ): Response<AuthResponse>

    @GET("/api/enrollment/status")
    suspend fun getEnrollmentStatus(@Header("Authorization") token: String): Response<EnrollmentStatusResponse>

    @GET("/api/enrollment/verify")
    suspend fun verifyEnrollment(@Header("Authorization") token: String): Response<VerificationResponse>
}
