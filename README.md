```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `mortality_monitor.service` to `/etc/systemd/system`
and run (as root) `systemctl enable mortality_monitor`.
