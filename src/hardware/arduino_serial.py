"""Nonblocking transport for the existing DIST:/RGB newline protocol.

Writes only confirm that bytes reached the serial driver; the firmware provides
no acknowledgement that the RGB LED changed.
"""

import logging
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional

import serial


LOGGER = logging.getLogger(__name__)
COMMANDS = frozenset(("GREEN", "YELLOW", "RED", "BLUE", "OFF"))
DISTANCE_PATTERN = re.compile(r"DIST:([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\Z")
SERIAL_ERRORS = (OSError, serial.SerialException)


def parse_distance(line: str, min_cm: float, max_cm: float) -> Optional[float]:
    """Parse a complete line, rejecting invalid or unsupported readings."""
    match = DISTANCE_PATTERN.fullmatch(line.strip())
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    if not math.isfinite(value) or value <= 0 or not min_cm <= value <= max_cm:
        return None
    return value


class DistanceTrigger:
    """Hysteresis with freshness; missing data always disables inspection."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._distance = None  # type: Optional[float]
        self._last_valid_at = None  # type: Optional[float]
        self._active = False

    def update(self, value: float, now: float) -> None:
        if (not math.isfinite(value) or value <= 0 or
                not self.config["min_distance_cm"] <= value <= self.config["max_distance_cm"]):
            return
        # A new reading after a gap starts a new hysteresis session.
        self.state(now)
        self._distance = value
        self._last_valid_at = now
        if value < self.config["enter_cm"]:
            self._active = True
        elif value > self.config["exit_cm"]:
            self._active = False

    def state(self, now: float) -> Dict[str, Any]:
        age = None if self._last_valid_at is None else max(0.0, now - self._last_valid_at)
        reading_state = "unavailable" if age is None else "fresh"
        if age is not None and age >= self.config["reading_timeout_s"]:
            reading_state = "stale"
            self._active = False
        return {
            "distance_cm": self._distance,
            "last_valid_age_s": age,
            "reading_state": reading_state,
            "active": self._active,
        }


class ArduinoSerial:
    """Bounded reads, nonblocking startup, retry scheduling and write caching."""

    def __init__(self, config: Dict[str, Any], factory: Optional[Callable[..., Any]] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.config = config
        self._factory = factory or serial.Serial
        self._clock = clock
        self._serial = None  # type: Optional[Any]
        self._next_retry_at = 0.0
        self._ready_at = 0.0
        self._buffer = bytearray()
        self._discarding_line = False
        self._last_command = None  # type: Optional[str]
        self.error = None  # type: Optional[str]
        # Changes on connect and disconnect, allowing the caller to reset fruit
        # and trigger state even when a reconnect happens between frame polls.
        self.generation = 0

    @property
    def connected(self) -> bool:
        return self._serial is not None

    @property
    def ready(self) -> bool:
        return self.connected and self._clock() >= self._ready_at

    def _disconnect(self) -> None:
        connection = self._serial
        self._serial = None
        self._buffer.clear()
        self._discarding_line = False
        self._last_command = None
        if connection is not None:
            self.generation += 1
            try:
                connection.close()
            except SERIAL_ERRORS as exc:
                LOGGER.warning("Serial close failed: %s", exc)

    def _failure(self, operation: str, exc: Exception) -> None:
        self.error = "Arduino serial {} failed: {}".format(operation, exc)
        LOGGER.error(self.error)
        self._disconnect()
        self._next_retry_at = self._clock() + self.config["retry_interval_s"]

    def _connect_if_due(self) -> None:
        if self.connected or self._clock() < self._next_retry_at:
            return
        try:
            connection = self._factory(
                port=self.config["port"], baudrate=self.config["baudrate"],
                timeout=0, write_timeout=self.config["write_timeout_s"],
            )
        except SERIAL_ERRORS as exc:
            self._failure("connection", exc)
            return
        self._serial = connection
        self._ready_at = self._clock() + self.config["startup_delay_s"]
        self._last_command = None
        self._buffer.clear()
        self._discarding_line = False
        self.error = None
        self.generation += 1
        LOGGER.info("Arduino port %s opened; startup delay %.2f seconds",
                    self.config["port"], self.config["startup_delay_s"])

    def poll(self) -> List[float]:
        """Read at most max_read_bytes; retain partial lines between frames."""
        self._connect_if_due()
        if not self.ready:
            return []
        try:
            available = self._serial.in_waiting
            if available <= 0:
                return []
            chunk = self._serial.read(min(available, self.config["max_read_bytes"]))
        except SERIAL_ERRORS as exc:
            self._failure("read", exc)
            return []
        values = []  # type: List[float]
        for byte in chunk:
            if byte == 10:  # Existing command/measurement framing uses LF.
                if not self._discarding_line:
                    try:
                        line = self._buffer.decode("ascii")
                    except UnicodeDecodeError:
                        line = ""
                    value = parse_distance(line, self.config["min_distance_cm"],
                                           self.config["max_distance_cm"])
                    if value is not None:
                        values.append(value)
                self._buffer.clear()
                self._discarding_line = False
            elif not self._discarding_line:
                if len(self._buffer) >= self.config["max_line_bytes"]:
                    self._buffer.clear()
                    self._discarding_line = True
                else:
                    self._buffer.append(byte)
        return values

    def send(self, command: str) -> bool:
        """Cache only complete successful writes, never failed attempts."""
        if command not in COMMANDS:
            raise ValueError("Unsupported Arduino command: {}".format(command))
        if not self.ready:
            return False
        if command == self._last_command:
            return True
        payload = (command + "\n").encode("ascii")
        try:
            written = self._serial.write(payload)
            if written != len(payload):
                raise OSError("Incomplete serial write ({}/{})".format(written, len(payload)))
        except SERIAL_ERRORS as exc:
            self._failure("write", exc)
            return False
        self._last_command = command
        LOGGER.info("Arduino command written: %s (no hardware acknowledgement)", command)
        return True

    def close(self, normal: bool = True) -> None:
        """Normal exit attempts OFF; fatal exit attempts BLUE and preserves it."""
        if self.connected:
            command = "OFF" if normal else "BLUE"
            if not self.send(command):
                LOGGER.warning("Shutdown %s could not be written; port unavailable or starting", command)
        self._disconnect()
