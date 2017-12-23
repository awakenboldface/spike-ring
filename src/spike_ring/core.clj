(ns spike-ring.core
  (:gen-class)
  (:require [spike-ring.server :refer [server]]
            [mount.core :as mount]))

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (mount/start)
  (println "Hello, World!"))
