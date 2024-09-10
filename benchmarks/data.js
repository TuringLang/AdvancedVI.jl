window.BENCHMARK_DATA = {
  "lastUpdate": 1725966635710,
  "repoUrl": "https://github.com/TuringLang/AdvancedVI.jl",
  "entries": {
    "Benchmark Results": [
      {
        "commit": {
          "author": {
            "name": "TuringLang",
            "username": "TuringLang"
          },
          "committer": {
            "name": "TuringLang",
            "username": "TuringLang"
          },
          "id": "ad54e6a1a7b84dfe8c51213ab6828ad32829a4e0",
          "message": "Grant push permissions to Benchmark.yml",
          "timestamp": "2024-09-10T10:48:20Z",
          "url": "https://github.com/TuringLang/AdvancedVI.jl/pull/90/commits/ad54e6a1a7b84dfe8c51213ab6828ad32829a4e0"
        },
        "date": 1725966632920,
        "tool": "julia",
        "benches": [
          {
            "name": "normal + bijector/meanfield/ForwardDiff",
            "value": 481297154,
            "unit": "ns",
            "extra": "gctime=26307364\nmemory=816851152\nallocs=2729782\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "normal + bijector/meanfield/ReverseDiff",
            "value": 182849751.5,
            "unit": "ns",
            "extra": "gctime=7990868.5\nmemory=98515216\nallocs=2029783\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      }
    ]
  }
}