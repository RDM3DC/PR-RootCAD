# Adaptive Resistance Principle – Site

Public-facing website & knowledge hub for the Adaptive Resistance Principle (ARP).

## Goals
- Concise landing explaining ARP
- Concept & research pages in Markdown
- Expandable blog / updates channel
- Fast static delivery (Astro)

## Commands
```bash
npm install
npm run dev
npm run build
npm run preview
```

## Deployment
- Recommended: Vercel (zero-config) or Netlify.
- GitHub Pages supported via `.github/workflows/deploy.yml`.

## Content Model
`src/content` holds Markdown/MDX pages (concept, research). Use frontmatter fields:
```yaml
title: "Title"
description: "Short meta description"
pubDate: 2025-01-01
draft: false
```

## Roadmap
- [ ] Add analytics (Plausible)
- [ ] Add email capture (Buttondown / Resend + serverless)
- [ ] Add blog taxonomy (tags)