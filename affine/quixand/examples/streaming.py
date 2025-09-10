#!/usr/bin/env python3
"""Interactive PTY streaming example with concurrent send/receive."""

from affine import quixand as qs
import threading
import time
import queue

def output_reader(term, stop_event):
    """Thread function to continuously read and print output from PTY."""
    print("\n📤 Output reader started")
    print("=" * 50)
    
    for chunk in term.stream():
        if stop_event.is_set():
            break
        
        text = chunk.decode("utf-8", errors="ignore")
        if text.strip():  # Only print non-empty output
            print(text, end="", flush=True)
    
    print("\n" + "=" * 50)
    print("📤 Output reader stopped")

def command_sender(term, commands):
    """Thread function to send commands with delays."""
    print("📥 Command sender started\n")
    
    for cmd_info in commands:
        cmd = cmd_info["cmd"]
        delay = cmd_info.get("delay", 1.0)
        
        print(f"\n>>> Sending: {cmd.strip()}")
        term.send(cmd)
        time.sleep(delay)  # Wait before sending next command
    
    print("\n📥 Command sender finished")

def main():
    print("=== Interactive PTY Streaming Example ===")
    print("This example demonstrates concurrent command sending and output streaming.\n")
    
    # Create a sandbox
    with qs.Sandbox(template="python:3.11-slim") as sandbox:
        print(f"✅ Sandbox created: {sandbox.id[:8]}")
        
        # Start an interactive PTY session
        with sandbox.pty("/bin/bash") as term:
            # Event to signal threads to stop
            stop_event = threading.Event()
            
            # Start output reader thread
            reader_thread = threading.Thread(
                target=output_reader, 
                args=(term, stop_event),
                daemon=True
            )
            reader_thread.start()
            
            # Define commands to execute
            commands = [
                {
                    "cmd": "echo 'Hello from sandbox!'",
                    "delay": 1.0
                },
                {
                    "cmd": "python3 -c \"import sys; print(f'Python version: {sys.version.split()[0]}')\"",
                    "delay": 2.0
                },
                {
                    "cmd": "for i in 1 2 3; do echo \"Count: $i\"; sleep 0.5; done",
                    "delay": 3.0
                },
                {
                    "cmd": "ls -la | head -5",
                    "delay": 1.0
                },
                {
                    "cmd": "echo 'All commands completed!'",
                    "delay": 1.0
                },
                {
                    "cmd": "exit",
                    "delay": 0.5
                }
            ]
            
            # Start command sender thread
            sender_thread = threading.Thread(
                target=command_sender,
                args=(term, commands),
                daemon=True
            )
            sender_thread.start()
            
            # Wait for sender to finish
            sender_thread.join()
            
            # Give some time for final output
            time.sleep(1)
            
            # Signal reader to stop
            stop_event.set()
            
            # Wait for reader thread with timeout
            reader_thread.join(timeout=2)
        
        print("\n✅ PTY session closed")
    
    print("✅ Streaming example completed successfully!")


if __name__ == "__main__":
    main()
