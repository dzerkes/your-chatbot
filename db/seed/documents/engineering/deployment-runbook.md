---
title: Deployment runbook
groups: engineering
tags: engineering, operations
owner: Platform team
updated: 2026-10-20
---

# Deployment runbook

How we ship changes to client environments. Follow it for every production
release, including hotfixes.

## Before you deploy

1. The change is merged to `main` and CI is green.
2. Release notes are written in the project's changelog.
3. Database migrations have been run against a copy of production data in
   staging, and the downgrade path was tested.
4. The client's change window is open. Most clients allow releases from Monday
   to Thursday, 09:00 to 16:00 their time. Harbor Logistics allows releases
   only on Tuesdays and Thursdays because of warehouse shift changes.

## Deploying

Start the release from the deployment pipeline, never from a laptop. The
pipeline deploys to one canary instance first and waits ten minutes. Watch the
error rate and p95 latency on the project dashboard during that time.

If both stay within 10 percent of the previous day, promote the release to all
instances. Otherwise roll back.

```bash
# Roll back to the previous release from the pipeline runner
deployctl rollback --project "$PROJECT" --to previous
```

## After you deploy

Post the release notes in the client channel within one hour. Keep watching
the dashboard for 30 minutes and stay reachable for two hours. Close the
change ticket only after the client confirms.

## Hotfixes

A hotfix may skip the change window if the client's on-call contact approves it
in writing. It still goes through CI and the canary step. Write a short
incident note within two working days that explains why the hotfix was needed.
