import socket
import os

def tcp_server(host='127.0.0.1', port=9001):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f'Server listening on {host}:{port}')
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            # Send the file upon connection.
            file_path = 'model.pt'
            file_size = os.path.getsize(file_path)
            
            # Open the file in binary mode
            with open(file_path, 'rb') as f:
                bytes_read = f.read(file_size)
                conn.sendall(bytes_read)
            
            print(f"Total bytes sent: {file_size}")
            print("File sent successfully.")

if __name__ == "__main__":
    tcp_server()
