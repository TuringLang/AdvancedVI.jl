window.BENCHMARK_DATA = {
  "lastUpdate": 1718245024783,
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
      },
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
          "id": "75eb334a31a4ff1036501d1f9d8d8ed4b327fde6",
          "message": "remove signature with user-defined restructure (#64)",
          "timestamp": "2024-06-07T01:21:07+01:00",
          "tree_id": "c6b799c4dff0b474b04759534fce8930d5bad550",
          "url": "https://github.com/TuringLang/AdvancedVI.jl/commit/75eb334a31a4ff1036501d1f9d8d8ed4b327fde6"
        },
        "date": 1717719818784,
        "tool": "julia",
        "benches": [
          {
            "name": "normal + bijector/meanfield/ForwardDiff",
            "value": 498137471,
            "unit": "ns",
            "extra": "gctime=31567906\nmemory=851982592\nallocs=3233671\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "normal + bijector/meanfield/ReverseDiff",
            "value": 141700614.5,
            "unit": "ns",
            "extra": "gctime=8378886.5\nmemory=89566656\nallocs=1899672\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
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
          "id": "5ced9c2dae5bb7c78b005a0b4684afc0def9b76b",
          "message": "add indirection for update step, add projection for `LocationScale` (#65)\n\n* add indirection for update step, add projection for `LocationScale`\r\n* add projection for `Bijectors` with `MvLocationScale`",
          "timestamp": "2024-06-13T01:48:32+01:00",
          "tree_id": "24250832e7d38a3adf8f5bf445cab91d9255cc43",
          "url": "https://github.com/TuringLang/AdvancedVI.jl/commit/5ced9c2dae5bb7c78b005a0b4684afc0def9b76b"
        },
        "date": 1718239862313,
        "tool": "julia",
        "benches": [
          {
            "name": "normal + bijector/meanfield/ForwardDiff",
            "value": 535462845,
            "unit": "ns",
            "extra": "gctime=28419173.5\nmemory=861934528\nallocs=3387669\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "normal + bijector/meanfield/ReverseDiff",
            "value": 192925746.5,
            "unit": "ns",
            "extra": "gctime=7137156.5\nmemory=98894592\nallocs=2053670\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "msca8h@naver.com",
            "name": "Ray Kim",
            "username": "Red-Portal"
          },
          "committer": {
            "email": "msca8h@naver.com",
            "name": "Ray Kim",
            "username": "Red-Portal"
          },
          "distinct": true,
          "id": "d44e36df2b25d4ab154aa1c44e8ddb89475e2ccc",
          "message": "add test group options",
          "timestamp": "2024-06-13T02:19:42+01:00",
          "tree_id": "d35ab17ed1c6b648f87b59816e589c4e51fc484d",
          "url": "https://github.com/TuringLang/AdvancedVI.jl/commit/d44e36df2b25d4ab154aa1c44e8ddb89475e2ccc"
        },
        "date": 1718241783545,
        "tool": "julia",
        "benches": [
          {
            "name": "normal + bijector/meanfield/ForwardDiff",
            "value": 558668726,
            "unit": "ns",
            "extra": "gctime=31767432.5\nmemory=861934528\nallocs=3387669\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "normal + bijector/meanfield/ReverseDiff",
            "value": 190422991,
            "unit": "ns",
            "extra": "gctime=7982450\nmemory=98894592\nallocs=2053670\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "msca8h@naver.com",
            "name": "Ray Kim",
            "username": "Red-Portal"
          },
          "committer": {
            "email": "msca8h@naver.com",
            "name": "Ray Kim",
            "username": "Red-Portal"
          },
          "distinct": true,
          "id": "b927e806b55f29c686eb2aaa57d39491572c8555",
          "message": "fix reduce computational cost of tests, use more sophisticated tests",
          "timestamp": "2024-06-13T03:14:09+01:00",
          "tree_id": "6091968cd8cb18c1a8d5383f9462a452532c27ea",
          "url": "https://github.com/TuringLang/AdvancedVI.jl/commit/b927e806b55f29c686eb2aaa57d39491572c8555"
        },
        "date": 1718245023906,
        "tool": "julia",
        "benches": [
          {
            "name": "normal + bijector/meanfield/ForwardDiff",
            "value": 558006515.5,
            "unit": "ns",
            "extra": "gctime=32716098.5\nmemory=861934528\nallocs=3387669\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "normal + bijector/meanfield/ReverseDiff",
            "value": 189274695,
            "unit": "ns",
            "extra": "gctime=7994632\nmemory=98894592\nallocs=2053670\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      }
    ]
  }
}