package com.voice.shield

import android.content.Intent
import android.os.Bundle
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

class CallHistoryActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()
        setContentView(R.layout.activity_call_history)

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

        // Open call details
        findViewById<android.view.View>(R.id.card_call_1).setOnClickListener {
            val intent = Intent(this, CallDetailsActivity::class.java)
            startActivity(intent)
        }

        findViewById<android.view.View>(R.id.card_call_2).setOnClickListener {
            val intent = Intent(this, CallDetailsActivity::class.java)
            startActivity(intent)
        }

        findViewById<android.view.View>(R.id.card_call_3).setOnClickListener {
            val intent = Intent(this, CallDetailsActivity::class.java)
            startActivity(intent)
        }

        // Bottom navigation
        findViewById<android.view.View>(R.id.nav_home).setOnClickListener {
            val intent = Intent(this, DashboardActivity::class.java)
            startActivity(intent)
            finish()
        }
    }
}