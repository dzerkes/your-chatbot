---
title: Incident response guide
groups: engineering
tags: engineering, operations, on-call
owner: Platform team
updated: 2026-08-05
---

# Incident response guide

What to do when a client system is down or degraded.

## Severity levels

- **SEV1**: the system is unavailable or data is at risk. Page the on-call
  engineer and the delivery lead immediately, day or night.
- **SEV2**: a major feature is broken or performance is badly degraded for many
  users. Respond within 30 minutes during business hours and within two hours
  outside them.
- **SEV3**: a minor problem with a workaround. Handle it in normal working
  hours and track it as a ticket.

## Roles

Every SEV1 and SEV2 incident has an incident lead, who coordinates and makes
decisions, and a communications lead, who updates the client every 30 minutes.
One person must not hold both roles for more than an hour.

## During an incident

Open an incident channel named after the ticket. Write down what you observe
and every action you take, with times. Prefer rolling back a recent release
over fixing forward. If the incident may involve personal data, tell the
Security team at once, as the data handling policy requires.

## After an incident

Hold a blameless review within five working days for every SEV1 and SEV2. The
review records the timeline, the root cause, what went well, and follow-up
actions with owners and dates. Share the summary with the client if the
contract requires it.
