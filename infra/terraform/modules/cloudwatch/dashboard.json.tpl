{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "title": "Prediction Volume",
        "metrics": [["${namespace}", "PredictionCount"]],
        "period": 300,
        "stat": "Sum",
        "region": "${region}",
        "view": "timeSeries"
      }
    },
    {
      "type": "metric",
      "properties": {
        "title": "P99 Prediction Latency (ms)",
        "metrics": [["${namespace}", "PredictionLatency", {"stat": "p99"}]],
        "period": 300,
        "region": "${region}",
        "view": "timeSeries"
      }
    },
    {
      "type": "metric",
      "properties": {
        "title": "Default Probability Distribution",
        "metrics": [["${namespace}", "DefaultProbability", {"stat": "Average"}]],
        "period": 300,
        "region": "${region}",
        "view": "timeSeries"
      }
    },
    {
      "type": "metric",
      "properties": {
        "title": "Feature PSI",
        "metrics": [["${namespace}", "PSI"]],
        "period": 3600,
        "stat": "Maximum",
        "region": "${region}",
        "view": "timeSeries"
      }
    },
    {
      "type": "metric",
      "properties": {
        "title": "Live AUC",
        "metrics": [["${namespace}", "LiveAUC"]],
        "period": 3600,
        "stat": "Average",
        "region": "${region}",
        "view": "timeSeries",
        "annotations": {
          "horizontal": [{"value": 0.75, "label": "Threshold", "color": "#ff0000"}]
        }
      }
    },
    {
      "type": "metric",
      "properties": {
        "title": "ECS CPU Utilization",
        "metrics": [["AWS/ECS", "CPUUtilization", "ClusterName", "${ecs_cluster}", "ServiceName", "${ecs_service}"]],
        "period": 300,
        "stat": "Average",
        "region": "${region}",
        "view": "timeSeries"
      }
    },
    {
      "type": "metric",
      "properties": {
        "title": "ECS Memory Utilization",
        "metrics": [["AWS/ECS", "MemoryUtilization", "ClusterName", "${ecs_cluster}", "ServiceName", "${ecs_service}"]],
        "period": 300,
        "stat": "Average",
        "region": "${region}",
        "view": "timeSeries"
      }
    },
    {
      "type": "metric",
      "properties": {
        "title": "ALB 5xx Error Rate",
        "metrics": [
          ["AWS/ApplicationELB", "HTTPCode_Target_5XX_Count", "LoadBalancer", "${alb_suffix}", "TargetGroup", "${tg_suffix}"],
          ["AWS/ApplicationELB", "RequestCount", "LoadBalancer", "${alb_suffix}", "TargetGroup", "${tg_suffix}"]
        ],
        "period": 60,
        "stat": "Sum",
        "region": "${region}",
        "view": "timeSeries"
      }
    }
  ]
}
