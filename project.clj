(defproject spike-ring "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url  "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [ring "1.6.3"]
                 [ring/ring-defaults "0.3.1"]
                 [ring/ring-json "0.4.0"]
                 [fogus/ring-edn "0.3.0"]
                 [clj-http "3.7.0"]
                 [aid "0.1.1"]
                 [macchiato/core "0.2.2"]
                 [org.clojure/clojurescript "1.9.946"]
                 [com.rpl/specter "1.0.5"]
                 [org.immutant/web "2.1.9"]
                 [mount "0.1.11"]]
  :plugins [[lein-ancient "0.6.10"]
            [lein-cljsbuild "1.1.7"]
            [lein-auto "0.1.3"]
            [lein-npm "0.6.2"]]
  :main ^:skip-aot spike-ring.core
  :uberjar-name "spike-ring.jar"
  :target-path "target/%s"
  :npm {:dependencies [[en-inflectors "1.0.12"]
                       [http-shutdown "1.2.0"]]}
  :cljsbuild {:builds [{:id           "prod"
                        :source-paths ["src"]
                        :compiler     {:main           spike-ring.server
                                       :output-to      "target/index.js"
                                       :target         :nodejs
                                       :output-dir     "target"
                                       ;:install-deps   true
                                       ;:npm-deps       {:en-inflectors "1.0.12"}
                                       ;; :externs ["externs.js"]
                                       ;:optimizations  :advanced
                                       :pretty-print   true
                                       :parallel-build true}}]}
  :profiles {:uberjar {:aot :all}})
