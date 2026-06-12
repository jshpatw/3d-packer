import os
import json
import http.server
import socketserver
import webbrowser
import threading

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class SageDashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        # API endpoint to dynamically retrieve the GLB files in results
        if self.path == '/api/files':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Scan directories
            scan_folders = [
                "results/01_Safe_Packs",
                "results/04_Frontier_Research",
                "results/05_Generalized_Tests",
                "VISUALIZATION_ASSETS"
            ]
            
            file_map = {}
            for folder in scan_folders:
                folder_path = os.path.join(DIRECTORY, folder)
                file_map[folder] = []
                if os.path.exists(folder_path):
                    for file in sorted(os.listdir(folder_path)):
                        if file.endswith('.glb') or file.endswith('.gltf'):
                            file_map[folder].append({
                                "name": file,
                                "path": f"/{folder}/{file}"
                            })
                            
            self.wfile.write(json.dumps(file_map).encode('utf-8'))
            return
            
        # Redirect root url to the visualizer index page
        if self.path == '/' or self.path == '/index.html':
            self.path = '/visualizer/index.html'
            
        return super().do_GET()

def open_browser():
    webbrowser.open(f"http://localhost:{PORT}")

def main():
    # Force socket reuse to avoid "address already in use" errors during quick restarts
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), SageDashboardHandler) as httpd:
        print(f"\n=======================================================")
        print(f"   SAGE 3D Packing Visualizer Dashboard is starting")
        print(f"   URL: http://localhost:{PORT}")
        print(f"   Root directory: {DIRECTORY}")
        print(f"=======================================================\n")
        
        # Open browser in a separate thread after server starts
        threading.Timer(1.0, open_browser).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down visualizer server.")

if __name__ == "__main__":
    main()
