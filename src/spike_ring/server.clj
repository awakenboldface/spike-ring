(ns spike-ring.server
  (:require [immutant.web :as web]))

(defn app [request]
  {:status 200
   :body   "Hello world!"})

;(web/run app)
