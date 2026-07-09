import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/chat': 'http://api:8001',
      '/health': 'http://api:8001',
      '/upload': 'http://api:8001',
      '/documents': 'http://api:8001',
    }
  }
})
