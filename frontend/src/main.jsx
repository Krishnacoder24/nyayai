import { StrictMode } from 'react' // strict mode development mode, helps to find potential problems in an application. It activates additional checks and warnings for its descendants. It does not render any visible UI.
import { createRoot } from 'react-dom/client' // createRoot is a method from the react-dom/client package that creates a root for rendering a React application. It is used to initialize the rendering process and manage the lifecycle of the application.
import App from './App.jsx'
import './index.css'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
