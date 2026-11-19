# ITE abstract draft

**Title**

Distinguishing a missing freeway detector reading from emerging congestion

**Track**

Transportation Operations

Traffic operations staff rely on neighboring detectors to tell an instrumentation problem from a queue. We built a freeway replay tool that makes this comparison explicit. It aligns five-minute detector readings by travel direction, retains the last good reading and its age, and displays a baseline and a graph-based reconstruction when a reading is missing. It recommends checking the detector when adjacent stations remain clear, and corroborating a queue when adjacent stations measure concurrent low speeds. Missing evidence is shown as unavailable.

We exercised the workflow on a Caltrans PeMS corridor with an injected detector outage and a separate observed congestion window. The missing reading prompted a detector check, and the observed congestion prompted a queue check. The replay retained the withheld reading for scoring and reproduced the revised output by rerunning only the windows that depended on that input. These examples show how a traffic operations team could inspect a proposed action and trace it back to the available readings. Broader alert accuracy and recovery time will be tested across independent operating periods.
