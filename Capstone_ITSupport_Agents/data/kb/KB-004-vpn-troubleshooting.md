---
id: KB-004
title: VPN (GlobalConnect) connection problems
category: vpn
updated: 2026-09-03
---
# VPN (GlobalConnect) connection problems

Bayview uses the GlobalConnect VPN client. Portal address: `vpn.bayview.example`.

## Quick checks
1. Confirm your internet connection works without VPN (open any public website).
2. Check the IT status page or ask the IT Assistant whether the VPN service has a known outage.
3. Make sure the portal address is exactly `vpn.bayview.example`.

## Error messages
- **"Authentication failed"**: your password may have expired or changed. Reset it (KB-001) and reconnect. Repeated failures can lock your account (KB-002).
- **"Gateway not responding" / stuck on "Connecting"**: quit GlobalConnect fully (right-click tray icon > Quit), then reopen it. If you are on hotel or airport Wi-Fi, accept the captive portal page in a browser first.
- **"Certificate error"**: your laptop's device certificate may have expired. Restart the laptop while connected to the office network or run **Company Portal > Sync**. If it persists, contact the Service Desk.
- **Connected but cannot reach internal sites**: disconnect, run `ipconfig /flushdns` in Command Prompt, reconnect.

## Still not working?
Collect the GlobalConnect logs (Settings > Troubleshooting > Collect Logs) and contact the Service Desk with the time of the failure.
