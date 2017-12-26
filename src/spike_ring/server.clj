(ns spike-ring.server
  (:require [immutant.web :as web]
            [ring.middleware.defaults :refer :all]
            [ring.middleware.json :refer [wrap-json-response wrap-json-body]]
            [mount.core :refer [defstate]]))

(defn app
  [request]
  (println (:body request))
  {:status 200
   :body   [{:hello "world!"} {:hello "world!"}]})

;(web/run (wrap-defaults app api-defaults))
(def start
  (partial web/run
           (-> app
               (wrap-json-body {:keywords? true})
               wrap-json-response)
           {:host "0.0.0.0"}))

(def stop
  (partial web/stop))

(defstate server
          :start (start)
          :stop (stop))
