// Minimal service worker: registration + passthrough fetch.
// Chrome's PWA install criteria need a non-trivial fetch handler;
// passthrough satisfies that without racing the access-cookie set
// during page load (precache of authed URLs would 401 here).
self.addEventListener("install", e => self.skipWaiting());
self.addEventListener("activate", e => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", e => e.respondWith(fetch(e.request)));
