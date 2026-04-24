import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import config
import time
import threading
import cv2

class NotificationManager:
    def __init__(self):
        self.last_sent = 0
        self.cooldown = config.NOTIFICATION_COOLDOWN
        self.alarm_start_time = None
        
    def check_and_alert(self, is_alarm_active, frame=None):
        """Called every frame. If alarm is active for X seconds, send alert (optionally with snapshot)."""
        if not is_alarm_active:
            if self.alarm_start_time is not None:
                print("[Alert] Alarm state cleared — timer reset.")
            self.alarm_start_time = None
            return
            
        # Alarm is active...
        if self.alarm_start_time is None:
            self.alarm_start_time = time.time()
            print("[Alert] Alarm started — waiting for 5s continuous...")
            return
            
        elapsed = time.time() - self.alarm_start_time
        
        # If alarm persistent for > 5 seconds
        if elapsed > config.ALARM_DURATION_THRESHOLD:
            # Check cooldown
            if time.time() - self.last_sent > self.cooldown:
                print(f"!!! SENDING EMERGENCY ALERT (alarm active for {elapsed:.1f}s) !!!")
                photo_bytes = None
                if frame is not None:
                    try:
                        ok, buf = cv2.imencode(".jpg", frame)
                        if ok:
                            photo_bytes = buf.tobytes()
                    except Exception as e:
                        print(f"Snapshot encode failed: {e}")

                self.send_alert_async(photo_bytes=photo_bytes)
                self.last_sent = time.time()
                
    def send_alert_async(self, photo_bytes=None):
        t = threading.Thread(target=self._send_alert_thread, args=(photo_bytes,))
        t.daemon = True
        t.start()
        
    def get_location(self):
        try:
            # Get location based on IP address
            response = requests.get('http://ip-api.com/json/')
            data = response.json()
            if data['status'] == 'success':
                lat = data['lat']
                lon = data['lon']
                city = data['city']
                region = data['regionName']
                country = data['country']
                maps_link = f"https://www.google.com/maps?q={lat},{lon}"
                return f"Location: {city}, {region}, {country}\nGoogle Maps: {maps_link}"
            return "Location: Unable to determine (API error)"
        except Exception as e:
            print(f"Location fetching failed: {e}")
            return "Location: Could not fetch location data."

    def _send_alert_thread(self, photo_bytes):
        msg = "URGENT: Driver Drowsiness Detected! Driver is not responding to alarm.\n"
        
        # Fetch current location
        location_info = self.get_location()
        msg += f"\n{location_info}"

        # 1. Telegram
        if config.USE_TELEGRAM:
            try:
                if photo_bytes:
                    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendPhoto"
                    data = {
                        "chat_id": config.TELEGRAM_CHAT_ID,
                        "caption": msg,
                    }
                    files = {"photo": ("drowsy.jpg", photo_bytes)}
                    resp = requests.post(url, data=data, files=files, timeout=15)
                    print(f"Telegram Photo Alert Response: {resp.status_code} - {resp.text}")
                else:
                    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
                    payload = {
                        "chat_id": config.TELEGRAM_CHAT_ID,
                        "text": msg
                    }
                    resp = requests.post(url, json=payload, timeout=15)
                    print(f"Telegram Text Alert Response: {resp.status_code} - {resp.text}")
            except Exception as e:
                print(f"Telegram Failed: {e}")
                
        # 2. Email / SMS Gateway
        if config.USE_EMAIL:
            try:
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(config.EMAIL_SENDER, config.EMAIL_PASSWORD)
                
                email_msg = MIMEMultipart()
                email_msg['From'] = config.EMAIL_SENDER
                email_msg['To'] = config.EMAIL_RECIPIENT
                email_msg['Subject'] = "DRIVER ALERT"
                email_msg.attach(MIMEText(msg, 'plain'))
                
                server.send_message(email_msg)
                server.quit()
                print("Email/SMS Alert Sent.")
            except Exception as e:
                print(f"Email Failed: {e}")

# Global instance
notifier = NotificationManager()
