"""Code that runs on a rented box, not on the validator host.

The files here are copied to the box as they are. They import the standard
library only, they hold no credential, and they reach R2 only through the
URLs that the host signed for one lease (``job.json``).
"""
