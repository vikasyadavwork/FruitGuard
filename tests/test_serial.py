import unittest

from src.hardware.arduino_serial import ArduinoSerial, DistanceTrigger, parse_distance


CONFIG = {
    "port": "COM5", "baudrate": 9600, "startup_delay_s": 2.0,
    "retry_interval_s": 3.0, "write_timeout_s": 0.1,
    "max_read_bytes": 4096, "max_line_bytes": 128,
    "min_distance_cm": 0.1, "max_distance_cm": 400.0,
    "enter_cm": 40.0, "exit_cm": 45.0, "reading_timeout_s": 1.5,
}


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeSerial:
    def __init__(self):
        self.input = bytearray()
        self.writes = []
        self.closed = False
        self.fail_write = False
        self.fail_read = False
        self.partial_write = False

    @property
    def in_waiting(self):
        if self.fail_read:
            raise OSError("disconnected")
        return len(self.input)

    def read(self, size):
        chunk = bytes(self.input[:size])
        del self.input[:size]
        return chunk

    def write(self, payload):
        if self.fail_write:
            raise OSError("disconnected")
        self.writes.append(payload)
        return len(payload) - 1 if self.partial_write else len(payload)

    def close(self):
        self.closed = True


class DistanceTests(unittest.TestCase):
    def test_valid_and_invalid_lines(self):
        self.assertEqual(parse_distance("DIST:42.35\r\n", 0.1, 400), 42.35)
        self.assertEqual(parse_distance("DIST:4.2e1", 0.1, 400), 42)
        for line in ("DIST:", "DIST:NaN", "DIST:inf", "DIST:1e999", "DIST:-1",
                     "DIST:0", "DIST:401", "DIST:0.01", "DIST:12:13", "noise",
                     "DIST:1_0", "DIST: 40", "DIST:42 junk"):
            with self.subTest(line=line):
                self.assertIsNone(parse_distance(line, 0.1, 400))

    def test_hysteresis_boundaries_staleness_and_reset(self):
        trigger = DistanceTrigger(CONFIG)
        self.assertEqual(trigger.state(0)["reading_state"], "unavailable")
        trigger.update(40, 0)
        self.assertFalse(trigger.state(0)["active"])
        trigger.update(39, 0.1)
        self.assertTrue(trigger.state(0.1)["active"])
        trigger.update(45, 0.2)
        self.assertTrue(trigger.state(0.2)["active"])
        trigger.update(46, 0.3)
        self.assertFalse(trigger.state(0.3)["active"])
        trigger.update(39, 1)
        self.assertEqual(trigger.state(2.5)["reading_state"], "stale")
        self.assertFalse(trigger.state(2.5)["active"])
        # The old measured distance remains explicitly stale, never invented.
        self.assertEqual(trigger.state(2.5)["distance_cm"], 39)
        trigger.update(42, 3)
        self.assertFalse(trigger.state(3)["active"])
        trigger.reset()
        self.assertIsNone(trigger.state(3)["distance_cm"])

    def test_invalid_reading_does_not_refresh_age(self):
        trigger = DistanceTrigger(CONFIG)
        trigger.update(30, 0)
        for value in (float("nan"), float("inf"), 0, -10, 401):
            trigger.update(value, 1)
        self.assertEqual(trigger.state(1.5)["reading_state"], "stale")


class SerialTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.ports = []
        self.factory_kwargs = []

        def factory(**kwargs):
            self.factory_kwargs.append(kwargs)
            port = FakeSerial()
            self.ports.append(port)
            return port

        self.transport = ArduinoSerial(CONFIG, factory, self.clock)

    def ready_transport(self):
        self.transport.poll()
        self.clock.now = 2
        self.transport.poll()
        return self.ports[-1]

    def test_nonblocking_startup_and_partial_lines(self):
        self.assertEqual(self.transport.poll(), [])
        self.assertTrue(self.transport.connected)
        self.assertFalse(self.transport.ready)
        self.assertFalse(self.transport.send("GREEN"))
        self.assertEqual(self.factory_kwargs[0]["timeout"], 0)
        self.assertEqual(self.factory_kwargs[0]["write_timeout"], 0.1)
        port = self.ports[0]
        port.input.extend(b"DIST:42.")
        self.clock.now = 2
        self.assertEqual(self.transport.poll(), [])
        port.input.extend(b"35\nDIST:nan\nDIST:-1\nnoise\nDIST:35\r\n")
        self.assertEqual(self.transport.poll(), [42.35, 35])

    def test_bounded_read_and_overlong_line(self):
        port = self.ready_transport()
        self.transport.config = dict(CONFIG, max_read_bytes=16, max_line_bytes=12)
        port.input.extend(b"X" * 30 + b"\nDIST:20\n")
        self.assertEqual(self.transport.poll(), [])
        self.assertEqual(len(port.input), 23)
        self.assertEqual(self.transport.poll(), [])
        self.assertEqual(self.transport.poll(), [20])

    def test_dedup_failed_write_and_reconnect(self):
        port = self.ready_transport()
        self.assertTrue(self.transport.send("GREEN"))
        self.assertTrue(self.transport.send("GREEN"))
        self.assertEqual(port.writes, [b"GREEN\n"])
        generation = self.transport.generation
        port.fail_write = True
        self.assertFalse(self.transport.send("RED"))
        self.assertFalse(self.transport.connected)
        self.assertTrue(port.closed)
        self.assertIn("write failed", self.transport.error)
        self.assertGreater(self.transport.generation, generation)
        self.clock.now = 4.9
        self.transport.poll()
        self.assertEqual(len(self.ports), 1)
        self.clock.now = 5
        self.transport.poll()
        self.assertEqual(len(self.ports), 2)
        self.assertIsNone(self.transport.error)
        self.clock.now = 7
        self.assertTrue(self.transport.send("GREEN"))
        self.assertTrue(self.transport.send("RED"))
        self.assertEqual(self.ports[1].writes, [b"GREEN\n", b"RED\n"])

    def test_partial_write_is_failure(self):
        port = self.ready_transport()
        port.partial_write = True
        self.assertFalse(self.transport.send("RED"))
        self.assertFalse(self.transport.connected)
        self.assertIn("Incomplete", self.transport.error)

    def test_read_failure_drops_buffer_and_allows_retry(self):
        port = self.ready_transport()
        port.input.extend(b"DIST:2")
        self.assertEqual(self.transport.poll(), [])
        port.fail_read = True
        self.assertEqual(self.transport.poll(), [])
        self.assertFalse(self.transport.connected)
        self.clock.now = 5
        self.transport.poll()
        self.clock.now = 7
        self.ports[1].input.extend(b"0\nDIST:22\n")
        self.assertEqual(self.transport.poll(), [22])

    def test_unavailable_port_retries_only_when_due(self):
        calls = []

        def unavailable(**kwargs):
            calls.append(kwargs)
            raise OSError("port missing")

        transport = ArduinoSerial(CONFIG, unavailable, self.clock)
        transport.poll()
        transport.poll()
        self.assertEqual(len(calls), 1)
        self.clock.now = 3
        transport.poll()
        self.assertEqual(len(calls), 2)

    def test_shutdown_normal_and_fatal_commands(self):
        port = self.ready_transport()
        self.transport.close()
        self.assertEqual(port.writes, [b"OFF\n"])
        self.assertTrue(port.closed)
        self.transport.poll()
        self.clock.now = 4
        fatal_port = self.ports[-1]
        self.transport.close(normal=False)
        self.assertEqual(fatal_port.writes, [b"BLUE\n"])
        self.assertNotIn(b"OFF\n", fatal_port.writes)

    def test_invalid_command_rejected(self):
        with self.assertRaises(ValueError):
            self.transport.send("PURPLE")


if __name__ == "__main__":
    unittest.main()
