/**
 * Minimal static server with smart port selection (avoids EADDRINUSE).
 * Usage: node server.js
 */
const express = require("express");
const path = require("path");

const app = express();
app.use(express.static(path.join(__dirname)));

// Health
app.get("/healthz", (req,res)=>res.json({ ok:true }));

const start = async (port=3000) => {
  const srv = app.listen(port, () => {
    console.log(`[swingalyze] listening on http://localhost:${port}`);
  });
  srv.on("error", (err) => {
    if (err.code === "EADDRINUSE") {
      console.log(`[swingalyze] port ${port} in use; trying ${port+1}…`);
      start(port+1);
    } else {
      console.error(err);
      process.exit(1);
    }
  });
};
start();