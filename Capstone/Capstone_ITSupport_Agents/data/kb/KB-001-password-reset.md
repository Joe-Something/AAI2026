---
id: KB-001
title: How to reset your network password
category: password_reset
updated: 2026-08-14
---
# How to reset your network password

Your network password is used for Windows sign-in, email, VPN and the SSO portal. Passwords expire every 90 days.

## Self-service reset (recommended)
1. Go to the self-service portal at https://passwordreset.bayview.example and click **Forgot password**.
2. Enter your username (the part of your email before the @).
3. Approve the MFA prompt on your registered phone (Microsoft Authenticator push or SMS code).
4. Choose a new password and sign out of all devices, then sign back in.

## Password rules
- At least 14 characters; passphrases such as `harbor-lantern-quiet-47` are encouraged.
- Cannot reuse any of your last 10 passwords.
- Must not contain your name or username.

## After resetting
- Update the saved password on your phone's mail app and on the Wi-Fi profile `Bayview-Corp`, or your account may lock again from repeated old-password attempts.
- VPN uses the same password; reconnect after the change.

## When to contact the Service Desk
- You are not enrolled in MFA and cannot complete the self-service reset (see KB-003).
- You receive "account locked" after resetting (see KB-002).
The Service Desk will never ask you for your password.
