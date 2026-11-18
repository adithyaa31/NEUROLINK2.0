import React from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App.jsx';
import './styles.css';

class ErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { error: null }; }
  static getDerivedStateFromError(error) { return { error }; }
  componentDidCatch(error, info) { console.error('NEUROLINK render error', error, info); }
  render() {
    if (this.state.error) return <main style={{minHeight:'100vh',background:'#080D1B',color:'#F4F7FF',fontFamily:'Inter,system-ui',padding:'12vh 8vw'}}>
      <div style={{maxWidth:650,margin:'auto',border:'1px solid #293a58',background:'#101A30',padding:32,borderRadius:12}}>
        <div style={{color:'#00E5FF',fontSize:12,letterSpacing:2}}>NEUROLINK 2.0 · RECOVERY</div>
        <h1 style={{fontFamily:'Space Grotesk,system-ui'}}>The workspace hit a rendering error.</h1>
        <p style={{color:'#A8B4CC',lineHeight:1.7}}>The application could not draw this screen. Reload to try again. If it persists, share the browser console error with your developer.</p>
        <button onClick={()=>window.location.reload()} style={{background:'#167eaa',color:'white',border:0,borderRadius:6,padding:'11px 16px',cursor:'pointer'}}>Reload workspace</button>
        {import.meta.env.DEV&&<pre style={{whiteSpace:'pre-wrap',color:'#ffb6b6',fontSize:12,marginTop:22}}>{this.state.error?.message}</pre>}
      </div>
    </main>;
    return this.props.children;
  }
}

const root = document.getElementById('root');
if (!root) throw new Error('Missing #root mount element in index.html');
createRoot(root).render(<React.StrictMode><ErrorBoundary><BrowserRouter><App/></BrowserRouter></ErrorBoundary></React.StrictMode>);
