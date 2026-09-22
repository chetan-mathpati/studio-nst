# System Design

```text
                    Studio NST
                        |
          +-------------+-------------+
          |             |             |
       Gatys         Johnson         AdaIN
          |             |             |
          +-------------+-------------+
                        |
                 Experiment Runner
                        |
             +----------+----------+
             |          |          |
          Metrics    Artifacts   Metadata
             |          |          |
             +----------+----------+
                        |
                 Research Report
```

The repository separates:

- reusable image/model utilities,
- model implementations,
- experiment scripts,
- benchmark collection,
- documentation,
- generated outputs.
