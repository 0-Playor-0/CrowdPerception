// Single seam for the API's location. Change this one constant to point
// the whole frontend at a different host -- nothing else in static/js
// hardcodes a URL.
const API_BASE_URL = window.location.origin;
const WS_BASE_URL = API_BASE_URL.replace(/^http/, "ws");
