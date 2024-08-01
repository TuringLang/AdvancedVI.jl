window.BENCHMARK_DATA = {
  "lastUpdate": 1722518225863,
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
          "id": "ec4db7e7480c43330b0c8d06d0f2d65841b862b5",
          "message": "fix bug in test",
          "timestamp": "2024-06-13T03:22:52+01:00",
          "tree_id": "6ff0ad61ba7c5b1124278c423af73853ab281afd",
          "url": "https://github.com/TuringLang/AdvancedVI.jl/commit/ec4db7e7480c43330b0c8d06d0f2d65841b862b5"
        },
        "date": 1718245678904,
        "tool": "julia",
        "benches": [
          {
            "name": "normal + bijector/meanfield/ForwardDiff",
            "value": 537087137.5,
            "unit": "ns",
            "extra": "gctime=26687253.5\nmemory=861934528\nallocs=3387669\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "normal + bijector/meanfield/ReverseDiff",
            "value": 189969735,
            "unit": "ns",
            "extra": "gctime=7276601\nmemory=98894592\nallocs=2053670\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
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
          "id": "9062431dad05a819a9c105ad7ba826d142f34d08",
          "message": "fix use words instead of greek letters",
          "timestamp": "2024-06-13T03:32:42+01:00",
          "tree_id": "f11b57bb4463e25450a03d84b37a09bef5bb79c6",
          "url": "https://github.com/TuringLang/AdvancedVI.jl/commit/9062431dad05a819a9c105ad7ba826d142f34d08"
        },
        "date": 1718246125762,
        "tool": "julia",
        "benches": [
          {
            "name": "normal + bijector/meanfield/ForwardDiff",
            "value": 565275016.5,
            "unit": "ns",
            "extra": "gctime=30350402\nmemory=861934528\nallocs=3387669\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "normal + bijector/meanfield/ReverseDiff",
            "value": 195508939,
            "unit": "ns",
            "extra": "gctime=8269472\nmemory=98894592\nallocs=2053670\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
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
          "id": "c93b5d71cf025aacbdf94d83467fd8a218eac1e3",
          "message": "add comment for `entropy` on `LocationScale`",
          "timestamp": "2024-06-13T03:55:12+01:00",
          "tree_id": "f8be87990ca35dc7ade1644662451727f6f8400f",
          "url": "https://github.com/TuringLang/AdvancedVI.jl/commit/c93b5d71cf025aacbdf94d83467fd8a218eac1e3"
        },
        "date": 1718247479487,
        "tool": "julia",
        "benches": [
          {
            "name": "normal + bijector/meanfield/ForwardDiff",
            "value": 529221525.5,
            "unit": "ns",
            "extra": "gctime=25323331.5\nmemory=861934528\nallocs=3387669\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "normal + bijector/meanfield/ReverseDiff",
            "value": 187259407,
            "unit": "ns",
            "extra": "gctime=7058396\nmemory=98894592\nallocs=2053670\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
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
          "id": "cb3b8380ad03ed8e84f7a8bc679cc076ba4ede2e",
          "message": "fix avoid re-defining the differentiation objective to support AD pre-compilation (#66)\n\n* update interface for objective initialization\r\n* improve `RepGradELBO` to not redefine AD forward path\r\n* add auxiliary argument to `value_and_gradient!`",
          "timestamp": "2024-06-15T01:27:18+01:00",
          "tree_id": "cc4b7ef89f7c04110c6b3d4942e71c3d3e3c62b9",
          "url": "https://github.com/TuringLang/AdvancedVI.jl/commit/cb3b8380ad03ed8e84f7a8bc679cc076ba4ede2e"
        },
        "date": 1718411389220,
        "tool": "julia",
        "benches": [
          {
            "name": "normal + bijector/meanfield/ForwardDiff",
            "value": 468762711,
            "unit": "ns",
            "extra": "gctime=23744277\nmemory=816894528\nallocs=2732669\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "normal + bijector/meanfield/ReverseDiff",
            "value": 185732693,
            "unit": "ns",
            "extra": "gctime=7398695\nmemory=98462592\nallocs=2030670\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "3279477+yebai@users.noreply.github.com",
            "name": "Hong Ge",
            "username": "yebai"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b25d572fd28567c9c5ea0be575d28489b29bb6a7",
          "message": "Create DocNav.yml",
          "timestamp": "2024-07-16T14:58:05+01:00",
          "tree_id": "f079970f8e867f6aa61c5b4313b2530a1a915ca9",
          "url": "https://github.com/TuringLang/AdvancedVI.jl/commit/b25d572fd28567c9c5ea0be575d28489b29bb6a7"
        },
        "date": 1721138437769,
        "tool": "julia",
        "benches": [
          {
            "name": "normal + bijector/meanfield/ForwardDiff",
            "value": 479588113,
            "unit": "ns",
            "extra": "gctime=24019708\nmemory=816942384\nallocs=2733667\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "normal + bijector/meanfield/ReverseDiff",
            "value": 180341322.5,
            "unit": "ns",
            "extra": "gctime=7403416.5\nmemory=98510448\nallocs=2031668\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "123811742+shravanngoswamii@users.noreply.github.com",
            "name": "Shravan Goswami",
            "username": "shravanngoswamii"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d4aed1d8bdac3d46dbb3a505e575c96fa957f556",
          "message": "Create Format.yml (#73)",
          "timestamp": "2024-08-01T14:14:30+01:00",
          "tree_id": "6226cc47b134859323bf3b5d24c89190bbf004e7",
          "url": "https://github.com/TuringLang/AdvancedVI.jl/commit/d4aed1d8bdac3d46dbb3a505e575c96fa957f556"
        },
        "date": 1722518225165,
        "tool": "julia",
        "benches": [
          {
            "name": "normal + bijector/meanfield/ForwardDiff",
            "value": 479736724,
            "unit": "ns",
            "extra": "gctime=28244728\nmemory=816847104\nallocs=2729712\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "normal + bijector/meanfield/ReverseDiff",
            "value": 179131994,
            "unit": "ns",
            "extra": "gctime=8341961\nmemory=98415168\nallocs=2027713\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      }
    ]
  }
}