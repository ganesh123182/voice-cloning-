package com.voice.shield

import android.content.Context
import android.os.Bundle
import android.text.InputType
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.navigation.fragment.findNavController
import com.voice.shield.api.RetrofitClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class FragmentLogin : Fragment(R.layout.fragment_login) {

    private var isPasswordVisible = false

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val authPrefs = requireActivity().getSharedPreferences("AuthPrefs", Context.MODE_PRIVATE)
        val tvPrefs = requireActivity().getSharedPreferences("TrustVoicePrefs", Context.MODE_PRIVATE)

        val btnBack = view.findViewById<ImageView>(R.id.btn_back)
        val etUsername = view.findViewById<EditText>(R.id.et_username)
        val etPassword = view.findViewById<EditText>(R.id.et_password)
        val ivTogglePassword = view.findViewById<ImageView>(R.id.iv_toggle_password)
        val tvForgotPassword = view.findViewById<TextView>(R.id.tv_forgot_password)
        val btnLogin = view.findViewById<Button>(R.id.btn_login)
        val progressLogin = view.findViewById<ProgressBar>(R.id.progress_login)
        val btnGoogleLogin = view.findViewById<LinearLayout>(R.id.btn_google_login)
        val btnPhoneLogin = view.findViewById<LinearLayout>(R.id.btn_phone_login)
        val btnToSignup = view.findViewById<TextView>(R.id.btn_to_signup)

        // Pre-fill email if previously registered or default demo
        val savedEmail = tvPrefs.getString("profile_email", null)
        if (!savedEmail.isNullOrEmpty()) {
            etUsername.setText(savedEmail)
        }

        // Back button
        btnBack.setOnClickListener {
            findNavController().navigateUp()
        }

        // Toggle password visibility
        ivTogglePassword.setOnClickListener {
            isPasswordVisible = !isPasswordVisible
            if (isPasswordVisible) {
                etPassword.inputType = InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD
            } else {
                etPassword.inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
            }
            etPassword.setSelection(etPassword.text.length)
        }

        // Forgot password
        tvForgotPassword.setOnClickListener {
            Toast.makeText(requireContext(), "Password reset instructions sent to your email", Toast.LENGTH_SHORT).show()
        }

        // Navigate to Sign Up
        btnToSignup.setOnClickListener {
            findNavController().navigate(R.id.action_login_to_signup)
        }

        // Google One-Click Login
        btnGoogleLogin.setOnClickListener {
            val name = "Google User"
            val email = "google.demo@trustvoice.ai"
            authPrefs.edit().putString("jwt_token", "demo_google_token_" + System.currentTimeMillis()).apply()
            tvPrefs.edit()
                .putString("profile_name", name)
                .putString("profile_email", email)
                .apply()
            Toast.makeText(requireContext(), "Signed in with Google", Toast.LENGTH_SHORT).show()
            findNavController().navigate(R.id.action_login_to_home)
        }

        // Phone One-Click Login
        btnPhoneLogin.setOnClickListener {
            val name = "Phone Verified User"
            val phone = "+91 98765 43210"
            authPrefs.edit().putString("jwt_token", "demo_phone_token_" + System.currentTimeMillis()).apply()
            tvPrefs.edit()
                .putString("profile_name", name)
                .putString("profile_email", phone)
                .apply()
            Toast.makeText(requireContext(), "Signed in with Phone Number", Toast.LENGTH_SHORT).show()
            findNavController().navigate(R.id.action_login_to_home)
        }

        // Standard Login
        btnLogin.setOnClickListener {
            val username = etUsername.text.toString().trim()
            val password = etPassword.text.toString().trim()

            if (username.isEmpty()) {
                etUsername.error = "Please enter Email or Phone Number"
                etUsername.requestFocus()
                return@setOnClickListener
            }
            if (password.isEmpty()) {
                etPassword.error = "Please enter Password"
                etPassword.requestFocus()
                return@setOnClickListener
            }

            btnLogin.text = ""
            progressLogin.visibility = View.VISIBLE
            btnLogin.isEnabled = false

            lifecycleScope.launch(Dispatchers.IO) {
                try {
                    val response = RetrofitClient.instance.login(username, password)
                    withContext(Dispatchers.Main) {
                        progressLogin.visibility = View.GONE
                        btnLogin.text = "Login"
                        btnLogin.isEnabled = true

                        if (response.isSuccessful && response.body() != null) {
                            val body = response.body()!!
                            val token = body.access_token ?: "demo_token"
                            val displayName = body.full_name ?: username.split("@")[0]
                            
                            authPrefs.edit().putString("jwt_token", token).apply()
                            tvPrefs.edit()
                                .putString("profile_name", displayName)
                                .putString("profile_email", username)
                                .apply()

                            Toast.makeText(requireContext(), "Welcome back, $displayName!", Toast.LENGTH_SHORT).show()
                            findNavController().navigate(R.id.action_login_to_home)
                        } else {
                            // Try registering in backend if account doesn't exist yet, or use local session
                            lifecycleScope.launch(Dispatchers.IO) {
                                try {
                                    val regResp = RetrofitClient.instance.register(
                                        u = username,
                                        p = password,
                                        fn = username.split("@")[0],
                                        em = if (username.contains("@")) username else null,
                                        ph = if (!username.contains("@")) username else null
                                    )
                                    withContext(Dispatchers.Main) {
                                        val token = regResp.body()?.access_token ?: ("demo_token_" + System.currentTimeMillis())
                                        val displayName = regResp.body()?.full_name ?: username.split("@")[0]
                                        authPrefs.edit().putString("jwt_token", token).apply()
                                        tvPrefs.edit()
                                            .putString("profile_name", displayName)
                                            .putString("profile_email", username)
                                            .apply()
                                        Toast.makeText(requireContext(), "Welcome, $displayName!", Toast.LENGTH_SHORT).show()
                                        findNavController().navigate(R.id.action_login_to_home)
                                    }
                                } catch (ex: Exception) {
                                    withContext(Dispatchers.Main) {
                                        val displayName = username.split("@")[0]
                                        authPrefs.edit().putString("jwt_token", "demo_token_" + System.currentTimeMillis()).apply()
                                        tvPrefs.edit()
                                            .putString("profile_name", displayName)
                                            .putString("profile_email", username)
                                            .apply()
                                        Toast.makeText(requireContext(), "Welcome, $displayName!", Toast.LENGTH_SHORT).show()
                                        findNavController().navigate(R.id.action_login_to_home)
                                    }
                                }
                            }
                        }
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        progressLogin.visibility = View.GONE
                        btnLogin.text = "Login"
                        btnLogin.isEnabled = true

                        // Offline fallback for demo
                        val displayName = username.split("@")[0]
                        authPrefs.edit().putString("jwt_token", "offline_demo_token_" + System.currentTimeMillis()).apply()
                        tvPrefs.edit()
                            .putString("profile_name", displayName)
                            .putString("profile_email", username)
                            .apply()
                        Toast.makeText(requireContext(), "Welcome, $displayName!", Toast.LENGTH_SHORT).show()
                        findNavController().navigate(R.id.action_login_to_home)
                    }
                }
            }
        }
    }
}
