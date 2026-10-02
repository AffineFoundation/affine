import socket
import threading
import unittest

from ops.provision_distributed_verifiers import wait_for_coordinator


class CoordinatorStartupTests(unittest.TestCase):
    def test_waits_for_delayed_listener(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
            ready = threading.Timer(.05, listener.listen)
            ready.start()
            try:
                wait_for_coordinator('127.0.0.1', port, timeout=2)
                listener.settimeout(1)
                connection, _ = listener.accept()
                connection.close()
            finally:
                ready.join()

    def test_closed_coordinator_times_out_before_worker_launch(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            with self.assertRaisesRegex(TimeoutError, 'no verifier workers started'):
                wait_for_coordinator('127.0.0.1', listener.getsockname()[1], timeout=.05)


if __name__ == '__main__':
    unittest.main()
