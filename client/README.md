# OrbitWatch client

Next.js app: 3D globe, before/after slider with the model's change mask, strictness slider, and a `/results` page.

```bash
pnpm install
pnpm dev          # http://localhost:3000, expects the API at http://localhost:8000
```

Set `NEXT_PUBLIC_API_URL` to point at a different API. Everything shown comes from the API; if it is not running the page says so and never shows made-up data.
