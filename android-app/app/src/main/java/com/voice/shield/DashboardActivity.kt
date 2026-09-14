package com.voice.shield

import android.content.Intent
import android.os.Bundle
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

class DashboardActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()
        setContentView(R.layout.activity_dashboard)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars =
                insets.getInsets(WindowInsetsCompat.Type.systemBars())

            v.setPadding(
                systemBars.left,
                systemBars.top,
                systemBars.right,
                systemBars.bottom
            )

            insets
        }

        // Current Call
        findViewById<android.view.View>(R.id.card_call).setOnClickListener {
            val intent = Intent(this, IncomingCallActivity::class.java)
            startActivity(intent)
        }

        // History
        findViewById<android.view.View>(R.id.nav_history).setOnClickListener {
            val intent = Intent(this, CallHistoryActivity::class.java)
            startActivity(intent)
        }
    }
}