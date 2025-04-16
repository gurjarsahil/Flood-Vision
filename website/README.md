# Flood Vision Web Interface

This directory contains the web interface for the Flood Vision flood prediction and analysis system. The interface provides interactive visualization of flood predictions, risk assessments, and analytical tools for decision-making.

## Overview

The Flood Vision web interface is designed to make complex flood prediction data accessible and actionable for both emergency responders and the general public. It presents machine learning predictions in an intuitive visual format with appropriate context and recommendations.

## Files Structure

```
website/
├── index.html           # Main HTML file with the interface structure
├── css/
│   └── style.css        # Main stylesheet for the interface
├── js/
│   ├── main.js          # Core functionality and initialization
│   ├── map.js           # Map visualization and GIS functionality
│   ├── predictions.js   # Flood prediction model integration
│   └── analysis.js      # Data analysis and visualization tools
└── img/                 # Image assets and sample visualizations
    ├── icons/           # UI icons and markers
    ├── samples/         # Sample flood images for demonstration
    └── overlays/        # Map overlays and flood visualization layers
```

## Key Features

### 1. Interactive Dashboard

The main dashboard provides:
- Current flood risk overview
- Recent analyses and predictions
- Quick access to tools and visualizations
- Notification center for alerts and updates

### 2. Flood Prediction Tool

The prediction interface allows users to:
- Input environmental parameters (rainfall, river levels, soil saturation)
- Upload satellite imagery for analysis
- View probability-based flood predictions
- See time-to-flood estimates and affected areas

### 3. Map Visualization

The map component features:
- Multi-layer visualization of flood risk zones
- Infrastructure overlays showing critical facilities
- Historical flood event comparison
- Evacuation route planning

### 4. Analysis Tools

Advanced tools for detailed analysis:
- Water level trend monitoring
- Historical data comparison
- "What-if" scenario modeling
- Impact assessment for communities and infrastructure

### 5. Report Generation

Users can generate:
- Detailed PDF reports for emergency response teams
- Community alert notifications
- Data exports for further analysis
- Visualizations for planning and communication

## Setup and Installation

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/flood-vision.git
   cd flood-vision/website
   ```

2. Start a local server:
   ```bash
   # Using Python's built-in server
   python -m http.server 8000
   
   # OR using Node.js with live-server
   npx live-server
   ```

3. Access the interface at `http://localhost:8000`

### Production Deployment

For production deployment:

1. Build the optimized assets:
   ```bash
   npm run build
   ```

2. Deploy the contents of the `dist/` directory to your web server

3. Ensure proper CORS configuration for API access

## Integration with Backend

The web interface connects to the Python backend via a RESTful API:

- **Prediction API**: `/api/predict` endpoint for model inference
- **Data API**: `/api/data` endpoints for retrieving historical and current data
- **Analysis API**: `/api/analyze` endpoints for generating statistics and analytics
- **Export API**: `/api/export` endpoint for generating reports and exports

API documentation is available in the `docs/api/` directory.

## Customization

### Theming

You can customize the appearance by:
1. Modifying variables in `css/variables.css`
2. Updating logo and branding in the `img/branding/` directory
3. Configuring color schemes in `js/config.js`

### Map Layers

To add custom map layers:
1. Add GeoJSON or raster files to `data/layers/`
2. Register layers in `js/map-config.js`
3. Define styling in `css/map-styles.css`

## Browser Compatibility

The interface is tested and optimized for:
- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+
- Mobile browsers (iOS Safari, Android Chrome)

## Responsive Design

The interface is fully responsive:
- Desktop: Full-featured dashboard with advanced visualizations
- Tablet: Optimized layouts with all core functionality
- Mobile: Focused interface for critical information and alerts

## Accessibility

Accessibility features include:
- WCAG 2.1 AA compliance
- Screen reader compatible elements
- Keyboard navigation support
- Color schemes designed for color vision deficiencies

## User Documentation

For detailed usage instructions, refer to:
- User guide: `docs/user-guide.pdf`
- Video tutorials: `docs/tutorials/`
- Feature documentation: `docs/features/`

## Developer Documentation

For developers looking to extend the interface:
- Component documentation: `docs/components/`
- API integration guide: `docs/api-integration.md`
- UI/UX design system: `docs/design-system.md`

## Contact

For questions about the web interface:
- Developer: Sahil Kumar
- LinkedIn: [linkedin.com/in/sahil-kumar-](https://www.linkedin.com/in/sahil-kumar-)
- Project Link: [github.com/yourusername/flood-vision](https://github.com/yourusername/flood-vision) 