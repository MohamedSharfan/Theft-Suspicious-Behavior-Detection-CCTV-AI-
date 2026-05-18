event_database = []


def store_event(event):
    event_database.append(event)


def get_recent_events(minutes=10):
    return event_database[-20:]
