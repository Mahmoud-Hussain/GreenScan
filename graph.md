flowchart TD

    A["AI Training Process"]

    A --> B["Monitor CPU Usage"]
    A --> C["Monitor RAM Usage"]
    A --> D["Monitor Network Usage"]
    A --> E["Monitor Runtime"]

    B --> F["Estimate CPU Power"]
    C --> G["Estimate RAM Power"]
    D --> H["Estimate Network Power"]

    F --> I["Calculate Total Power"]
    G --> I
    H --> I

    I --> J["Calculate Energy Consumption"]

    E --> J

    J --> K["Apply Carbon Intensity Factor"]

    K --> L["Calculate Carbon Emission"]

    L --> M["Store Metrics in Database"]

    M --> N["Dashboard and Analytics"]

    N --> O["Carbon Footprint Reports"]
    N --> P["Federated Learning Comparison"]
    N --> Q["Accuracy vs Energy Analysis"]
