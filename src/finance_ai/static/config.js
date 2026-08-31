/* Personal Finance AI — deployment config.
   Sets the backend API base URL for the frontend.

   Local dev: leave null → app.js falls back to the same origin
   (location.origin), so `make dev` works with zero changes.

   iHost deploy: point this at the hosted backend, e.g.
     window.FINANCE_API_BASE = "https://<app>.onrender.com";
*/
window.FINANCE_API_BASE = null;
