"""Distractor tool server for affine-notool-v1.

Every tool is a plausible assistant action (calendar, email, smart home,
travel, finance, files, media) that has nothing to do with the knowledge
question the task asks. Calling one is the mistake the source exists to
capture; the tool answers with a refusal that names the right move, so a
rollout that called a tool can still recover and answer in prose (its
`solved` grade is 0 either way — see the taskset).

Which tools a task offers is per task (`enabled` in the config, chosen by
the taskset from the question id), so the schemas vary like BFCL's
irrelevance items instead of one fixed menu the model could learn to
ignore.
"""

from __future__ import annotations

import verifiers.v1 as vf
from verifiers.v1.utils.decorators import discover_decorated

REFUSAL = ("Not applicable: this tool cannot help with a knowledge question. "
           "Answer from memory, in plain text, with no tool call.")


class NoToolToolsetConfig(vf.ToolsetConfig):
    enabled: list[str] = []
    """Tool names offered to the model (empty = every tool below)."""


class NoToolToolset(vf.Toolset[NoToolToolsetConfig]):
    # None = tools are advertised under their bare names (no server prefix).
    TOOL_PREFIX = None

    def register(self, mcp) -> None:
        want = set(self.config.enabled)
        for fn in discover_decorated(self, "tool"):
            name = getattr(fn, "tool_name", None) or fn.__name__
            if want and name not in want:
                continue
            mcp.add_tool(self._with_state(fn), name=name,
                         description=(fn.__doc__ or "").strip() or None)

    # -- calendar / communication ------------------------------------------------
    @vf.tool
    def create_calendar_event(self, title: str, start_time: str, end_time: str,
                              attendees: list[str] | None = None) -> str:
        """Create a calendar event with a title, ISO-8601 start and end times and optional attendee emails."""
        return REFUSAL

    @vf.tool
    def list_calendar_events(self, date: str) -> str:
        """List the calendar events on a given date (YYYY-MM-DD)."""
        return REFUSAL

    @vf.tool
    def send_email(self, to: str, subject: str, body: str) -> str:
        """Send an email to a recipient address with a subject line and body text."""
        return REFUSAL

    @vf.tool
    def send_sms(self, phone_number: str, message: str) -> str:
        """Send a text message to a phone number."""
        return REFUSAL

    @vf.tool
    def set_reminder(self, text: str, remind_at: str) -> str:
        """Set a reminder with the given text at an ISO-8601 time."""
        return REFUSAL

    @vf.tool
    def set_alarm(self, time: str, label: str = "") -> str:
        """Set an alarm for a time of day (HH:MM, 24-hour clock) with an optional label."""
        return REFUSAL

    # -- smart home ------------------------------------------------------------------
    @vf.tool
    def set_thermostat(self, temperature_celsius: float, mode: str = "auto") -> str:
        """Set the home thermostat target temperature in Celsius and the mode (heat, cool or auto)."""
        return REFUSAL

    @vf.tool
    def control_lights(self, room: str, state: str, brightness_percent: int = 100) -> str:
        """Turn the lights in a room on or off and set their brightness (0-100)."""
        return REFUSAL

    @vf.tool
    def lock_door(self, door: str, locked: bool) -> str:
        """Lock or unlock a named door of the house."""
        return REFUSAL

    @vf.tool
    def play_music(self, query: str, device: str = "living room") -> str:
        """Play music matching a search query on a named speaker."""
        return REFUSAL

    # -- travel / local -----------------------------------------------------------------
    @vf.tool
    def get_weather_forecast(self, city: str, days: int = 3) -> str:
        """Get the weather forecast for a city for the next N days."""
        return REFUSAL

    @vf.tool
    def search_flights(self, origin: str, destination: str, date: str,
                       passengers: int = 1) -> str:
        """Search flights between two airport codes on a date (YYYY-MM-DD) for a number of passengers."""
        return REFUSAL

    @vf.tool
    def book_hotel(self, city: str, check_in: str, check_out: str, guests: int = 1) -> str:
        """Book a hotel room in a city for the given check-in and check-out dates and number of guests."""
        return REFUSAL

    @vf.tool
    def request_ride(self, pickup: str, dropoff: str, ride_type: str = "standard") -> str:
        """Request a ride from a pickup address to a drop-off address."""
        return REFUSAL

    @vf.tool
    def find_restaurants(self, location: str, cuisine: str = "", max_price_level: int = 4) -> str:
        """Find restaurants near a location, optionally filtered by cuisine and price level (1-4)."""
        return REFUSAL

    @vf.tool
    def get_directions(self, origin: str, destination: str, mode: str = "driving") -> str:
        """Get turn-by-turn directions between two addresses by driving, walking, cycling or transit."""
        return REFUSAL

    # -- finance / shopping -------------------------------------------------------------
    @vf.tool
    def get_stock_quote(self, ticker: str) -> str:
        """Get the current market quote for a stock ticker symbol."""
        return REFUSAL

    @vf.tool
    def transfer_money(self, from_account: str, to_account: str, amount: float,
                       currency: str = "USD") -> str:
        """Transfer an amount of money between two of the user's bank accounts."""
        return REFUSAL

    @vf.tool
    def add_to_shopping_cart(self, product_id: str, quantity: int = 1) -> str:
        """Add a product to the user's shopping cart by product id."""
        return REFUSAL

    @vf.tool
    def track_package(self, tracking_number: str, carrier: str = "") -> str:
        """Track a parcel by its tracking number and optional carrier name."""
        return REFUSAL

    # -- files / dev ------------------------------------------------------------------------
    @vf.tool
    def create_file(self, path: str, content: str) -> str:
        """Create a file at a path with the given text content, overwriting any existing file."""
        return REFUSAL

    @vf.tool
    def list_directory(self, path: str) -> str:
        """List the entries of a directory on the user's workstation."""
        return REFUSAL

    @vf.tool
    def run_sql_query(self, database: str, query: str) -> str:
        """Run a SQL query against one of the user's named databases and return the rows."""
        return REFUSAL

    @vf.tool
    def create_github_issue(self, repository: str, title: str, body: str = "") -> str:
        """Open an issue in a GitHub repository (owner/name) with a title and body."""
        return REFUSAL


# Every @vf.tool above, by name; the taskset samples per-task subsets from
# this list (kept static so the catalog needs no server import).
TOOL_NAMES = (
    "create_calendar_event", "list_calendar_events", "send_email", "send_sms",
    "set_reminder", "set_alarm",
    "set_thermostat", "control_lights", "lock_door", "play_music",
    "get_weather_forecast", "search_flights", "book_hotel", "request_ride",
    "find_restaurants", "get_directions",
    "get_stock_quote", "transfer_money", "add_to_shopping_cart", "track_package",
    "create_file", "list_directory", "run_sql_query", "create_github_issue",
)


if __name__ == "__main__":
    NoToolToolset.run()
