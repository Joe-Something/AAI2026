---
id: KB-002
title: Account locked out
category: account_lockout
updated: 2026-07-02
---
# Account locked out

Accounts lock automatically after **5 failed sign-in attempts within 15 minutes**. A lockout lasts 30 minutes and then clears itself.

## Common causes
- A phone or tablet still using an old password for email (most common after a password change).
- A mapped network drive or saved credential in Windows Credential Manager with the old password.
- Typing the password with Caps Lock on.
- Someone else trying to sign in to your account. Repeated lockouts with no obvious cause should be reported as a security concern (see KB-011).

## Unlocking
- Wait 30 minutes, or request an unlock through the IT Assistant or Service Desk. Identity is verified with your MFA device before unlocking.
- Then remove stale saved passwords: Control Panel > Credential Manager > Windows Credentials, delete entries for `bayview` servers.

## Prevention
- Update your password on every device right after a reset.
- Use the Company Portal app on mobile instead of manually configured mail profiles.
