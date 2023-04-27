import http.server
import threading
import time
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# class meant to manage all ongoing actions, starts and stops actions
class MessageDispatcher:
    _actions = []

    # Decides whether or not the user is trying to start or stop an action and responds accordingly
    def dispatchMesassage(self, message):
        if message[0] == "S":
            self.closeAction(message[1:])

        else:
            newAction = MessageHandler(message)
            newAction.start()
            self._actions.append(newAction)

    # Searches for all actions that match the target that the user is trying to close such as slice or pan
    def closeAction(self, target):
        if target == "":
            self.closeAll()
        for action in self._actions:
            if action.message[1] == target:
                action.stop = True

    # Closes all ongoing functions
    def closeAll(self):
        for action in self._actions:
            action.stop = True


# Meant to handle the message ie print the message as many times as needed and return once it's done
# Every message handler runs on its own thread
class MessageHandler(threading.Thread):
    def __init__(self, message):
        self.message = message
        self.stop = False
        threading.Thread.__init__(self)

    def run(self):
        self.handle()
        return

    # Prints the actions the amount of times indicated in the message
    def handle(self):
        if self.message[0] == "C":
            action = self.message[1:]
            container = action.split()
            try:
                reps = int(container[1])
            except IndexError:
                eprint(
                    "Formatting error: amount of repetitions missing\n"
                    "the correct format for continuous actions is: " + self.message + " [number of times to repeat]"
                )
                return
            for i in range(reps):
                if self.stop:
                    break
                print(container[0])
                sys.stdout.flush()
                time.sleep(0.75)
        else:
            print(self.message)
            sys.stdout.flush()
        return


# In charge of recieving messages from AWS, passes all messages off to the message dispatcher
# Responds with ok to all messages, just as a tool for debugging
class HttpHandler(http.server.BaseHTTPRequestHandler):
    _dispatcher = MessageDispatcher()

    def setHeaders(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_POST(self):
        length = int(self.headers['Content-length'])
        self.setHeaders()
        self.end_headers()
        self.wfile.write(b"OK")
        self._dispatcher.dispatchMesassage(self.rfile.read(length).decode("utf-8"))

    def log_message(self, format, *args):
        return


# creates and runs the server, port is set to default to 80 and the handler is HttpHandler
def run(server_class=http.server.HTTPServer, http_handler=HttpHandler, port=80):
    server_address = ('', port)
    httpd = server_class(server_address, http_handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
        http_handler._dispatcher.closeAll()
        httpd.server_close()


if __name__ == "__main__":
    run()