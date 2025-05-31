from socketserver import TCPServer


class Server(TCPServer):
    allow_reuse_address = True
