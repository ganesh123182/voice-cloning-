import re

def update_retrofit():
    with open("android-app/app/src/main/java/com/voice/shield/api/RetrofitClient.kt", "r") as f:
        content = f.read()
    
    # We need to inject an AuthInterceptor that reads SharedPreferences
    # But RetrofitClient is an object. We can pass a context or just use an Application class.
    # Since we don't have Application class context easily available, we can make `instance` a function or pass the token.
    # Actually, a simple way is to pass token directly to the API methods using `@Header("Authorization")` in TrustVoiceApi.
    # Let's modify TrustVoiceApi.kt instead.
    pass

def patch_api():
    with open("android-app/app/src/main/java/com/voice/shield/api/TrustVoiceApi.kt", "r") as f:
        content = f.read()

    new_imports = """
import retrofit2.http.FormUrlEncoded
import retrofit2.http.Field
import retrofit2.http.Header
"""
    content = content.replace("import retrofit2.Response", "import retrofit2.Response\n" + new_imports)

    new_models = """
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
"""
    content = content.replace("interface TrustVoiceApi", new_models + "\ninterface TrustVoiceApi")

    # Update enrollVoice to take token and remove user_id from Part (now inferred from token)
    content = content.replace(
        "@Part(\"user_id\") userId: RequestBody,",
        "@Header(\"Authorization\") token: String,"
    )
    
    # Update EnrollmentResponse
    content = content.replace(
        "val hash: String?",
        "val evidence_hash: String?,\n    val blockchain_tx_hash: String?,\n    val enrollment_id: String?"
    )

    new_endpoints = """
    @FormUrlEncoded
    @POST("/api/login")
    suspend fun login(@Field("username") u: String, @Field("password") p: String): Response<TokenResponse>

    @FormUrlEncoded
    @POST("/api/register")
    suspend fun register(@Field("username") u: String, @Field("password") p: String): Response<GenericResponse>

    @GET("/api/enrollment/status")
    suspend fun getEnrollmentStatus(@Header("Authorization") token: String): Response<EnrollmentStatusResponse>

    @GET("/api/enrollment/verify")
    suspend fun verifyEnrollment(@Header("Authorization") token: String): Response<VerificationResponse>
"""
    # Insert before last brace
    content = content.rsplit("}", 1)[0] + new_endpoints + "}\n"

    with open("android-app/app/src/main/java/com/voice/shield/api/TrustVoiceApi.kt", "w") as f:
        f.write(content)

def patch_enrollment_fragment():
    with open("android-app/app/src/main/java/com/voice/shield/FragmentVoiceEnrollment.kt", "r") as f:
        content = f.read()

    # Need to update uploadAudio() to get the JWT token and pass it.
    upload_logic_old = """
                val requestFile = audioFile!!.asRequestBody("audio/wav".toMediaTypeOrNull())
                val body = MultipartBody.Part.createFormData("file", audioFile!!.name, requestFile)
                val userId = "user_123".toRequestBody("text/plain".toMediaTypeOrNull())

                val response = RetrofitClient.instance.enrollVoice(userId, body)
"""
    upload_logic_new = """
                val prefs = requireActivity().getSharedPreferences("AuthPrefs", android.content.Context.MODE_PRIVATE)
                val token = prefs.getString("jwt_token", null)
                if (token == null) {
                    withContext(Dispatchers.Main) { Toast.makeText(requireContext(), "Not Authenticated", Toast.LENGTH_SHORT).show() }
                    return@launch
                }
                
                val requestFile = audioFile!!.asRequestBody("audio/wav".toMediaTypeOrNull())
                val body = MultipartBody.Part.createFormData("file", audioFile!!.name, requestFile)
                
                val response = RetrofitClient.instance.enrollVoice("Bearer $token", body)
"""
    content = content.replace(upload_logic_old.strip(), upload_logic_new.strip())
    
    # Update Success Toast
    toast_old = "val hash = respBody?.hash?.take(8) ?: \"\"\n                        Toast.makeText(requireContext(), \"Enrolled on Blockchain! Hash: $hash...\", Toast.LENGTH_LONG).show()"
    toast_new = "val hash = respBody?.evidence_hash?.take(8) ?: \"\"\n                        val tx = respBody?.blockchain_tx_hash?.take(8) ?: \"N/A\"\n                        Toast.makeText(requireContext(), \"Enrolled! ID: ${respBody?.enrollment_id}\\nHash: $hash\\nTx: $tx\", Toast.LENGTH_LONG).show()"
    content = content.replace(toast_old, toast_new)

    with open("android-app/app/src/main/java/com/voice/shield/FragmentVoiceEnrollment.kt", "w") as f:
        f.write(content)

def patch_nav_graph():
    with open("android-app/app/src/main/res/navigation/nav_graph.xml", "r") as f:
        content = f.read()
    
    # Change start destination
    content = content.replace("app:startDestination=\"@id/nav_home\"", "app:startDestination=\"@id/nav_login\"")
    
    # Add Login Fragment
    login_fragment = """
    <fragment
        android:id="@+id/nav_login"
        android:name="com.voice.shield.FragmentLogin"
        android:label="Login"
        tools:layout="@layout/fragment_login">
        <action
            android:id="@+id/action_login_to_home"
            app:destination="@id/nav_home"
            app:popUpTo="@id/nav_login"
            app:popUpToInclusive="true"/>
    </fragment>
"""
    content = content.replace("</navigation>", login_fragment + "\n</navigation>")

    with open("android-app/app/src/main/res/navigation/nav_graph.xml", "w") as f:
        f.write(content)

if __name__ == "__main__":
    patch_api()
    patch_enrollment_fragment()
    patch_nav_graph()
    print("Android patches applied successfully.")
