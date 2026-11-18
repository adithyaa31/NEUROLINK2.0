const configured = import.meta.env.VITE_API_URL?.trim();
const base = (configured || 'http://127.0.0.1:8000').replace(/\/$/, '');
export async function api(path, options={}) {
  let response; const controller=new AbortController(); const timeout=setTimeout(()=>controller.abort(),5000);
  try {
    response=await fetch(base+path,{...options,headers:{'Content-Type':'application/json',...options.headers},signal:options.signal||controller.signal});
  } catch (error) {
    if (error.name === 'AbortError' || error.name === 'TimeoutError') throw new Error('The NEUROLINK API did not respond within 5 seconds. Check that FastAPI is running on port 8000.');
    throw new Error('Cannot reach the NEUROLINK API. Check that FastAPI is running on port 8000.');
  } finally { clearTimeout(timeout); }
  if(!response.ok) {
    let message=`API request failed (${response.status})`;
    try { const body=await response.json(); message=body.detail||message; } catch {}
    throw new Error(message);
  }
  return response.json();
}
export const apiBase=base;
