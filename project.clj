(defproject spike-ring "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url  "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [ring "1.6.3"]
                 [ring/ring-defaults "0.3.1"]
                 [ring/ring-json "0.4.0"]
                 [org.immutant/web "2.1.9"]
                 [mount "0.1.11"]]
  :plugins [[lein-ancient "0.6.10"]
            [lein-auto "0.1.3"]]
  :main ^:skip-aot spike-ring.core
  :uberjar-name "spike-ring.jar"
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
