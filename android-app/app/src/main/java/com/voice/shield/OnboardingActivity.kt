package com.voice.shield

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class OnboardingActivity : AppCompatActivity() {

    private lateinit var title: TextView
    private lateinit var description: TextView
    private lateinit var illustration: TextView
    private lateinit var nextButton: Button

    private lateinit var dot1: View
    private lateinit var dot2: View
    private lateinit var dot3: View

    private var currentPage = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_onboarding)

        title = findViewById(R.id.tv_title)
        description = findViewById(R.id.tv_description)
        illustration = findViewById(R.id.tv_illustration)
        nextButton = findViewById(R.id.btn_next)

        dot1 = findViewById(R.id.dot1)
        dot2 = findViewById(R.id.dot2)
        dot3 = findViewById(R.id.dot3)

        showPage(0)

        nextButton.setOnClickListener {

            if (currentPage < 2) {
                currentPage++
                showPage(currentPage)
            } else {

                // Temporary destination.
                // We will replace MainActivity with LoginActivity
                // after building the authentication screens.

                val intent = Intent(this, LoginActivity::class.java)
                startActivity(intent)
                finish()
            }
        }
    }

    private fun showPage(page: Int) {

        when (page) {

            0 -> {
                illustration.text = "🔊"
                title.text = "Stay Safe During Calls"
                description.text =
                    "TrustVoice checks voice signals in real time to help detect suspicious or cloned voices during calls."
                nextButton.text = "NEXT"
            }

            1 -> {
                illustration.text = "🛡️"
                title.text = "Know When a Voice Is Suspicious"
                description.text =
                    "Get a clear risk assessment when a call shows signs of voice impersonation or fraud."
                nextButton.text = "NEXT"
            }

            2 -> {
                illustration.text = "🔐"
                title.text = "Your Privacy Matters"
                description.text =
                    "TrustVoice is designed to protect your conversations while providing useful security evidence when needed."
                nextButton.text = "GET STARTED"
            }
        }

        updateDots(page)
    }

    private fun updateDots(page: Int) {

        val activeWidth = 24
        val inactiveWidth = 8

        dot1.layoutParams.width =
            if (page == 0) activeWidth else inactiveWidth

        dot2.layoutParams.width =
            if (page == 1) activeWidth else inactiveWidth

        dot3.layoutParams.width =
            if (page == 2) activeWidth else inactiveWidth

        dot1.requestLayout()
        dot2.requestLayout()
        dot3.requestLayout()
    }
}