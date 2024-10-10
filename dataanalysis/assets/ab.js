// Navigation toggle
document.addEventListener('DOMContentLoaded', () => {
    const navToggle = document.querySelector('.nav-toggle');
    const navMenu = document.querySelector('.nav-menu');

    navToggle?.addEventListener('click', () => {
        navMenu?.classList.toggle('active');
    });

    // Create all visualizations
    createDemographicsPlot();
    createModelPerformancePlot();
    createTimelinePlot();
});

// Demographics visualization
function createDemographicsPlot() {
    const data = [{
        type: 'bar',
        x: ['18-25', '26-35', '36-45', '46-55', '56+'],
        y: [15, 30, 25, 20, 10],
        name: 'Age Distribution',
        marker: {
            color: '#3498db'
        }
    }, {
        type: 'bar',
        x: ['Low', 'Medium', 'High'],
        y: [25, 45, 30],
        name: 'Income Segments',
        marker: {
            color: '#2ecc71'
        },
        visible: 'legendonly'
    }];

    const layout = {
        title: 'Customer Demographics Overview',
        barmode: 'group',
        responsive: true
    };

    Plotly.newPlot('demographicsPlot', data, layout);
}

// Model performance visualization
function createModelPerformancePlot() {
    const data = [{
        type: 'scatter',
        mode: 'lines+markers',
        x: ['Model 1', 'Model 2', 'Model 3'],
        y: [0.85, 0.92, 0.88],
        name: 'Accuracy',
        marker: {
            color: '#e74c3c'
        }
    }];

    const layout = {
        title: 'Predictive Models Performance',
        yaxis: {
            range: [0, 1],
            title: 'Accuracy Score'
        },
        responsive: true
    };

    Plotly.newPlot('modelPerformancePlot', data, layout);
}

// Implementation timeline visualization
function createTimelinePlot() {
    const data = [{
        type: 'scatter',
        mode: 'lines+markers',
        x: ['Month 1', 'Month 3', 'Month 6', 'Month 12'],
        y: [1, 2, 3, 4],
        text: ['Phase 1 Start', 'Phase 1 Complete', 'Phase 2 Complete', 'Phase 3 Complete'],
        marker: {
            size: 12
        }
    }];

    const layout = {
        title: 'Implementation Timeline',
        xaxis: {
            title: 'Timeline'
        },
        yaxis: {
            title: 'Implementation Progress',
            showticklabels: false
        },
        responsive: true,
        showlegend: false
    };

    Plotly.newPlot('timelinePlot', data, layout);
}

// Smooth scrolling for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});