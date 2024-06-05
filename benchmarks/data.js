window.BENCHMARK_DATA = {
  "lastUpdate": 1717630606412,
  "repoUrl": "https://github.com/TuringLang/AdvancedVI.jl",
  "entries": {
    "Benchmark Results": [
      {
        "commit": {
          "author": {
            "email": "msca8h@naver.com",
            "name": "Kyurae Kim",
            "username": "Red-Portal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "314eacf763af1c52153759268e30e87fbf17c5e7",
          "message": "add continuous benchmarking (#61)",
          "timestamp": "2024-06-06T00:34:12+01:00",
          "tree_id": "aaa9dc2e141f5d5744018e31585322976599e747",
          "url": "https://github.com/TuringLang/AdvancedVI.jl/commit/314eacf763af1c52153759268e30e87fbf17c5e7"
        },
        "date": 1717630605270,
        "tool": "julia",
        "benches": [
          {
            "name": "normal + bijector/meanfield/ForwardDiff",
            "value": 500583347,
            "unit": "ns",
            "extra": "gctime=31453527\nmemory=851662624\nallocs=3222678\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "normal + bijector/meanfield/ReverseDiff",
            "value": 136374500,
            "unit": "ns",
            "extra": "gctime=7583095\nmemory=89006704\nallocs=1881679\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      }
    ]
  }
}