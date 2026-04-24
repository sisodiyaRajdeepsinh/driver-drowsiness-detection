
# --- Notification Config ---

# Option 1: Email-to-SMS (Free, but need recipient carrier gateway)
# Common Gateways:
# Verizon: number@vtext.com
# AT&T: number@txt.att.net
# T-Mobile: number@tmomail.net
# Sprint: number@messaging.sprintpcs.com
USE_EMAIL = False
EMAIL_SENDER = "sisodiyarajdeep204@gmail.com"
EMAIL_PASSWORD = "your_app_password_here" # You need an App Password for Gmail
EMAIL_RECIPIENT = "recipient_number@carrier_gateway.com" 

# Option 2: Telegram Bot (Recommended, Free, Reliable)
# 1. Search for "BotFather" in Telegram
# 2. Type /newbot to create a bot and get TOKEN
# 3. Search for "userinfobot" to get your CHAT_ID
USE_TELEGRAM = True
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"  # Get from @BotFather on Telegram
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"       # Get from @userinfobot on Telegram

# Logic Config
ALARM_DURATION_THRESHOLD = 5 # Seconds of continuous alarm before sending alert
NOTIFICATION_COOLDOWN = 60 # Seconds to wait before sending another alert

# --- Accuracy Tuning ---
# Calibration: system records baseline EAR for this many frames at startup
CALIBRATION_FRAMES = 150  # More frames = more robust baseline under any lighting

# Mouth Aspect Ratio threshold for yawn detection
MAR_THRESH = 0.6

# Consecutive yawn frames required before counting as a real yawn
YAWN_CONSEC_FRAMES = 15

# Score increments / decrements (tuned to reduce false alarms under variable lighting)
SCORE_INC_EYES_CLOSED = 3   # Both EAR + CNN agree eyes are closed — high confidence
SCORE_INC_EAR_ONLY = 1      # Only EAR says closed (could be lighting noise)
SCORE_INC_CNN_ONLY = 1      # Only CNN says closed (could be lighting noise)
SCORE_INC_YAWN = 1           # Yawning detected
SCORE_DEC_NORMAL = 2         # Eyes open — moderate recovery speed
